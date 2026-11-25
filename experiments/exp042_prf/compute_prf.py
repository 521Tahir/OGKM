"""
exp042 - Accuracy, Precision, Recall and F1 for every model and arm.

These four were not previously computed; BERT-F1, METEOR, COMET, Key-token and
the LLM-judge Response Accuracy were. The free-form task has no single short gold
string, so "precision/recall/F1" has to be defined before it can be measured.
Two standard definitions are computed, both reported, because they answer
different questions:

  TOKEN-LEVEL (SQuAD-style)
      Content tokens of the prediction vs the gold answer, lowercased, stripped
      of punctuation and stopwords, compared as multisets.
        precision = overlap / |prediction tokens|
        recall    = overlap / |gold tokens|
        f1        = harmonic mean
      Macro-averaged over items. This is the usual generative-QA P/R/F1 and it
      rewards reproducing the reference's wording.

  ENTITY-LEVEL
      The same three quantities computed over gazetteer place names only (69
      curated Jeju locations with coordinates). This measures whether the answer
      names the right PLACES, independent of phrasing, which is what a tourism
      answer is actually judged on and what an ontology is supposed to supply.

  ACCURACY
      Free-form answers admit no exact match, so accuracy is reported as an
      item-level correctness rate at an explicit threshold:
        acc@token_f1>=0.5   fraction of items whose token F1 reaches 0.5
        acc@entity_f1>=0.5  fraction of items whose entity F1 reaches 0.5
        acc@entity_hit      fraction of items naming at least one gold place
      The judge-based Response Accuracy already in the scores directory is
      carried alongside for reference; it is a different measurement (an LLM
      grading the answer) and is not recomputed here.

Items where both prediction and gold contain no countable unit are skipped for
that metric family and the count is reported, so an empty denominator never
silently inflates a score.

  python -m experiments.exp042_prf.compute_prf --split test
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
GAZ = json.loads((ROOT / "data" / "jeju_full" / "jeju_gazetteer.json")
                 .read_text(encoding="utf-8"))
PLACES = [x["name"] for x in GAZ["locations"] if isinstance(x, dict)]
# longest-first so "Jeju Folk Village" wins over "Jeju"
PLACE_RE = re.compile("|".join(re.escape(p) for p in sorted(PLACES, key=len, reverse=True)),
                      re.IGNORECASE)

STOP = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "to", "from", "in", "on", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "you", "your", "we", "our",
    "i", "my", "can", "will", "would", "should", "may", "also", "as", "into",
    "out", "up", "down", "then", "than", "so", "such", "there", "here", "which",
    "who", "what", "when", "where", "how", "all", "any", "some", "more", "most",
    "other", "each", "do", "does", "did", "have", "has", "had", "not", "no",
}
TOKEN_RE = re.compile(r"[a-z0-9']+")

MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "deepseek-r1-8b",
          "qwen3-8b", "mistral-7b"]
ARMS = [("rag", "scores", "generations", "Simple RAG"),
        ("rag", "scores_ded", "generations_ded", "Simple RAG (dedicated)"),
        ("onto_v3", "scores", "generations", "Ontology V3"),
        ("onto_v3", "scores_ded", "generations_ded", "Ontology V3 (dedicated)"),
        ("onto_c", "scores", "generations", "Ontology (cleaned)"),
        ("hybrid20", "scores", "generations", "Onto-RAG hybrid 20%"),
        ("base", "scores", "generations", "Vanilla (no fine-tune)")]


def toks(s: str) -> Counter:
    return Counter(t for t in TOKEN_RE.findall((s or "").lower())
                   if t not in STOP and len(t) > 1)


def ents(s: str) -> Counter:
    return Counter(m.group(0).lower() for m in PLACE_RE.finditer(s or ""))


def prf(pred: Counter, gold: Counter):
    """Multiset precision/recall/F1. Returns None when nothing is countable."""
    if not pred and not gold:
        return None
    if not pred or not gold:
        return (0.0, 0.0, 0.0)
    ov = sum((pred & gold).values())
    p = ov / sum(pred.values())
    r = ov / sum(gold.values())
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return (p, r, f)


def judge_mean(score_dir: str, split: str, model: str, arm: str):
    p = OUT / score_dir / f"{split}__{model}__{arm}.jsonl"
    if not p.exists():
        return None
    v = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("judge"), (int, float)):
            v.append(r["judge"])
    return st.mean(v) if v else None


def evaluate(gen_dir: str, split: str, model: str, arm: str):
    p = OUT / gen_dir / f"{split}__{model}__{arm}.jsonl"
    if not p.exists():
        return None
    seen = set()
    tp, tr, tf, ep, er, ef = [], [], [], [], [], []
    hit = []
    n_tok_skip = n_ent_skip = 0
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = r.get("record_id")
        if rid in seen:
            continue
        seen.add(rid)
        gold = r.get("reference") or r.get("Output") or ""
        pred = r.get("answer") or r.get("prediction") or ""
        t = prf(toks(pred), toks(gold))
        if t is None:
            n_tok_skip += 1
        else:
            tp.append(t[0]); tr.append(t[1]); tf.append(t[2])
        ge, pe = ents(gold), ents(pred)
        e = prf(pe, ge)
        if e is None:
            n_ent_skip += 1
        else:
            ep.append(e[0]); er.append(e[1]); ef.append(e[2])
            hit.append(1.0 if sum((pe & ge).values()) > 0 else 0.0)
    if not tf:
        return None
    return {
        "n": len(seen),
        "token_precision": st.mean(tp), "token_recall": st.mean(tr),
        "token_f1": st.mean(tf),
        "entity_precision": st.mean(ep) if ep else None,
        "entity_recall": st.mean(er) if er else None,
        "entity_f1": st.mean(ef) if ef else None,
        "acc_token_f1_50": st.mean(1.0 if x >= .5 else 0.0 for x in tf),
        "acc_entity_f1_50": st.mean(1.0 if x >= .5 else 0.0 for x in ef) if ef else None,
        "acc_entity_hit": st.mean(hit) if hit else None,
        "items_without_entities": n_ent_skip,
    }


def fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v*100:.1f}" if pct else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="experiments/exp042_prf/prf_results.json")
    a = ap.parse_args()

    res: dict = {}
    for model in MODELS:
        for arm, sdir, gdir, label in ARMS:
            r = evaluate(gdir, a.split, model, arm)
            if not r:
                continue
            r["judge_response_accuracy"] = judge_mean(sdir, a.split, model, arm)
            res.setdefault(model, {})[label] = r
        if model in res:
            print(f"  {model}: {len(res[model])} arms", flush=True)

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"split": a.split, "results": res}, indent=2), encoding="utf-8")

    print(f"\n{'='*132}\nTOKEN-LEVEL (SQuAD-style, content tokens)   split = {a.split}\n{'='*132}")
    hdr = (f"{'model':16s} {'arm':26s} {'n':>6s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
           f"{'Acc@F1>=.5':>11s} {'JudgeAcc':>9s}")
    print(hdr); print("-" * len(hdr))
    for model in MODELS:
        if model not in res:
            continue
        for label, r in res[model].items():
            print(f"{model:16s} {label:26s} {r['n']:6,d} {fmt(r['token_precision']):>7s} "
                  f"{fmt(r['token_recall']):>7s} {fmt(r['token_f1']):>7s} "
                  f"{fmt(r['acc_token_f1_50']):>11s} {fmt(r['judge_response_accuracy']):>9s}")
        print()

    print(f"{'='*132}\nENTITY-LEVEL (gazetteer place names)        split = {a.split}\n{'='*132}")
    hdr = (f"{'model':16s} {'arm':26s} {'n':>6s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
           f"{'Acc@F1>=.5':>11s} {'Acc@hit':>8s} {'no-ent':>7s}")
    print(hdr); print("-" * len(hdr))
    for model in MODELS:
        if model not in res:
            continue
        for label, r in res[model].items():
            print(f"{model:16s} {label:26s} {r['n']:6,d} {fmt(r['entity_precision']):>7s} "
                  f"{fmt(r['entity_recall']):>7s} {fmt(r['entity_f1']):>7s} "
                  f"{fmt(r['acc_entity_f1_50']):>11s} {fmt(r['acc_entity_hit']):>8s} "
                  f"{r['items_without_entities']:7,d}")
        print()
    print(f"wrote {p.relative_to(ROOT)}")
    print("\nAll values are percentages except n and no-ent (item counts).")
    print("Acc@F1>=.5 = fraction of items reaching F1 0.50; Acc@hit = fraction naming "
          "at least one gold place.")


if __name__ == "__main__":
    main()
