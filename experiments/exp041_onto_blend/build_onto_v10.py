"""
exp041 - onto_v10: fixes a real defect found by auditing the context files.

The defect
----------
Measured on test_*.jsonl by comparing two different questions' contexts
byte-for-byte:

    arm        ctx chars   STATIC across questions   question-specific
    rag           2,913         11  ( 0.4%)               2,902
    onto_v3       2,990        238  ( 8.0%)               2,752
    onto_v6       3,032      1,983  (65.4%)               1,049   <-- defect
    onto_v9       3,043      1,393  (45.8%)               1,650   <-- defect

onto_v6 reached gold-entity coverage 0.512 (vs onto_v3's 0.270) by appending a
popularity list - but that list is built ONCE and reused verbatim for all 4,607
items. Two thirds of its budget is a constant prefix, so the model received only
1,049 question-relevant characters against Simple RAG's 2,902. The coverage number
was real; the retrieval was not.

onto_v3 is properly question-specific but ranks candidates by dense similarity to
the question, plus - in its class-routing step - by ONTOLOGY DEGREE. Neither
signal is corpus salience, which is what actually predicts the places a gold
itinerary names.

The fix
-------
Fold salience into the per-question ranking instead of appending it as a block:

    score(entity) = alpha * cosine(question, card) + (1 - alpha) * salience(entity)

salience comes from train-split mention frequency (exp033's entity_prior, rank-
normalised); candidates are the top-`pool` cards by cosine, re-ranked by the blend,
then packed until the 3,000-character budget is full. Every character is
question-conditioned and popularity-aware at once - which no previous arm was.

Fairness unchanged: same 3,000-char budget as every arm, salience computed from
splits/train.jsonl only, the same corpus the RAG arm indexes.

  python -m experiments.exp041_onto_blend.build_onto_v10 --splits train test bench
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import OntoV3ContextBuilder, _truncate
from src.utils.proctitle import set_proctitle

PRIOR = ROOT / "experiments" / "exp033_onto_prior" / "entity_prior.json"


def key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test", "bench"])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on question similarity; 1-alpha on corpus salience")
    ap.add_argument("--forced", type=int, default=200,
                    help="top-salience cards always included as candidates")
    ap.add_argument("--pool", type=int, default=240,
                    help="dense candidates re-ranked by the blend")
    ap.add_argument("--facts-per-entity", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # rank-normalised salience: 1.0 for the most-mentioned train entity -> 0.0
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    sal = {key(e): 1.0 - i / max(len(prior) - 1, 1) for i, (e, _) in enumerate(prior)}
    print(f"salience prior: {len(sal):,} entities (top: {prior[0][0]})")

    b = OntoV3ContextBuilder(device=a.device)
    b.build()
    b.load()
    cards = b._cards

    # Salience matching: a prior entry such as "Manjanggul Cave" must score the card
    # named "Manjanggul". Exact first, then containment either way, longest match wins.
    card_sal: dict[int, float] = {}
    for i, c in enumerate(cards):
        k = key(c["name"])
        if k in sal:
            card_sal[i] = sal[k]
            continue
        best = 0.0
        for pk, v in sal.items():
            if (k in pk or pk in k) and min(len(k), len(pk)) >= 5:
                best = max(best, v)
        if best:
            card_sal[i] = best
    print(f"entity cards: {len(cards):,}; with salience: {len(card_sal):,}")

    # The salient cards are ALWAYS candidates. Previously the pool was the dense
    # top-240 only, so for a generic question ("plan a 3-day trip") the popular
    # places were never considered and the blend just reshuffled whatever the
    # similarity search happened to return - coverage 0.241, below onto_v3's 0.252.
    forced = sorted(card_sal, key=lambda i: -card_sal[i])[: a.forced]
    print(f"forced salient candidates per question: {len(forced):,}")

    import numpy as np

    def contexts_for(questions: list[str]) -> list[str]:
        model = b._load_model()
        qe = model.encode([C.EMBED_QUERY_PREFIX + q for q in questions],
                          batch_size=128, convert_to_numpy=True,
                          normalize_embeddings=True,
                          show_progress_bar=False).astype("float32")
        sims, idxs = b._index.search(qe, a.pool)
        out = []
        for srow, irow in zip(sims, idxs):
            simof = {int(j): float(sc) for sc, j in zip(srow, irow) if j >= 0}
            pool = set(simof) | set(forced)
            cand = [(a.alpha * simof.get(j, 0.0)
                     + (1.0 - a.alpha) * card_sal.get(j, 0.0), cards[j]) for j in pool]
            cand.sort(key=lambda t: -t[0])
            lines, used, n = [], set(), 0
            for _, c in cand:
                if key(c["name"]) in used:
                    continue
                used.add(key(c["name"]))
                for f in c["facts"][: a.facts_per_entity]:
                    ln = f"- {f}"
                    if n + len(ln) + 1 > C.CTX_MAX_CHARS - 40:
                        break
                    lines.append(ln)
                    n += len(ln) + 1
                if n >= C.CTX_MAX_CHARS - 60:
                    break
            out.append("Knowledge-graph facts:\n" + "\n".join(lines))
        return out

    def read(split: str):
        p = {"train": C.TRAIN_JSONL, "val": C.VAL_JSONL,
             "test": C.TEST_JSONL, "bench": C.BENCH_JSONL}[split]
        if split == "bench":
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    for split in a.splits:
        recs = read(split)
        out = C.CTX_DIR / f"{split}_onto_v10.jsonl"
        chunk = 512
        with open(out, "w", encoding="utf-8") as f:
            for i in range(0, len(recs), chunk):
                part = recs[i:i + chunk]
                ctxs = contexts_for([r["Input"] for r in part])
                for r, ctx in zip(part, ctxs):
                    f.write(json.dumps({"record_id": r.get("record_id"),
                                        "Input": r["Input"], "Output": r["Output"],
                                        "context": _truncate(ctx, C.CTX_MAX_CHARS)},
                                       ensure_ascii=False) + "\n")
                if i % 5120 == 0:
                    print(f"  [{split}] {i:,}/{len(recs):,}", flush=True)
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        import statistics as stt
        # how much of the context is shared with a DIFFERENT question?
        x, y = rows[0]["context"], rows[min(100, len(rows) - 1)]["context"]
        j = 0
        while j < min(len(x), len(y)) and x[j] == y[j]:
            j += 1
        print(f"  {split}: {len(rows):,} -> {out.name}  "
              f"ctx {int(stt.mean(len(r['context']) for r in rows)):,} chars, "
              f"shared-prefix with another question {j} chars")


if __name__ == "__main__":
    main()
