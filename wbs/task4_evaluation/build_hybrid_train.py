"""
WBS Task 4.1 - training contexts for a dedicated V4/V5 adapter.

V2 and V3 are served by adapters trained on their own context format. V4 and V5
are served by `mixed`, an adapter trained on a blend, so the two arms that combine
ontology facts with retrieved prose are judged on weights that were never fitted
to the format they see. That is the mixed-adapter confound, and it sits on exactly
the arms most likely to beat Simple RAG.

Removing it needs a training file whose contexts are assembled by the same rule
`run_variants.py` uses at inference: the ontology context truncated to 20% of the
budget, the RAG context filling the rest, under the two sub-headers. This writes
that file.

The arm is named `hybrid20d` - a NEW key, not the existing `hybrid20`. The stored
`*_hybrid20.jsonl` files and the published exp036 numbers keep their own prompt
header; nothing here changes them.

  python -m wbs.task4_evaluation.build_hybrid_train --splits train val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FACT_SHARE = 0.20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.contexts import _truncate

    fact_budget = int(C.CTX_MAX_CHARS * FACT_SHARE)
    for split in a.splits:
        rp = C.CTX_DIR / f"{split}_rag.jsonl"
        op = C.CTX_DIR / f"{split}_onto_v3.jsonl"
        out = C.CTX_DIR / f"{split}_hybrid20d.jsonl"
        if not rp.exists() or not op.exists():
            print(f"[{split}] missing {rp.name if not rp.exists() else op.name} — skip",
                  flush=True)
            continue
        if out.exists() and not a.force:
            print(f"[{split}] {out.name} exists — skip", flush=True)
            continue
        n, mismatched, fc, pc = 0, 0, 0, 0
        with open(rp, encoding="utf-8") as fr, open(op, encoding="utf-8") as fo, \
                open(out, "w", encoding="utf-8") as fw:
            for lr, lo in zip(fr, fo):
                r, o = json.loads(lr), json.loads(lo)
                if r.get("record_id") != o.get("record_id"):
                    # the two files are written from the same split in the same
                    # order; if that ever stops being true the merge is garbage
                    mismatched += 1
                    continue
                oc = o["context"]
                facts = _truncate(oc if not oc.startswith("[no ontology") else "",
                                  fact_budget)
                passages = _truncate(r["context"], C.CTX_MAX_CHARS - len(facts))
                ctx = ((f"Ontology facts:\n{facts}\n\n" if facts else "")
                       + f"Retrieved reference passages:\n{passages}")
                fw.write(json.dumps({"record_id": r.get("record_id"),
                                     "Input": r["Input"], "Output": r["Output"],
                                     "context": ctx, "fact_chars": len(facts),
                                     "passage_chars": len(passages),
                                     "n_hits": r.get("n_hits", 0)
                                     + o.get("n_hits", 0)}, ensure_ascii=False) + "\n")
                n += 1; fc += len(facts); pc += len(passages)
        if mismatched:
            sys.exit(f"[{split}] {mismatched} record_id mismatches between "
                     f"{rp.name} and {op.name} — files are not aligned, refusing "
                     f"to ship this training set")
        print(f"[{split}] wrote {out.name}: {n:,} examples, "
              f"{fc // max(n, 1):,} fact chars + {pc // max(n, 1):,} passage chars "
              f"per example", flush=True)


if __name__ == "__main__":
    main()
