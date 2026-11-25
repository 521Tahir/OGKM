"""
exp032 - the hybrid arm: ontology-augmented RAG.

Motivation. Ontology-alone cannot win the free-form task: measured gold-entity
coverage plateaus at ~0.19 against Simple RAG's 0.726 across four independent
attempts (fact cleaning, entity-card retrieval, 2-hop graph traversal, and a full
rebuild of the ontology from all 36,847 training answers). The reason is
structural - a free-form itinerary question does not determine WHICH places the
gold answer names, so retrieving a comparable itinerary beats reasoning from
facts about places.

But ontology-alone was never the project's proposal. README.md defines the system
as "V4 Onto-RAG (hybrid)": retrieval AUGMENTED by the ontology. This arm builds
that, and compares it against Simple RAG on equal terms.

Fairness. The hybrid gets the SAME total character budget as Simple RAG
(CTX_MAX_CHARS). It is a reallocation, not an enlargement: passages are trimmed
to make room for facts, so the comparison isolates whether structured facts are
worth some of the passage budget - not whether more context helps.

  python -m experiments.exp032_hybrid.build_hybrid --splits test bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from experiments.exp024_rag_vs_ontology.contexts import _truncate
from src.utils.proctitle import set_proctitle

# Share of the common budget given to ontology facts; the rest goes to passages.
FACT_SHARE = 0.30


def load(split: str, arm: str) -> dict:
    p = C.CTX_DIR / f"{split}_{arm}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p} - build that arm's contexts first")
    return {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "bench"])
    ap.add_argument("--onto-arm", default="onto_v3",
                    help="which ontology arm supplies the facts (v3 has the best coverage)")
    ap.add_argument("--fact-share", type=float, default=FACT_SHARE)
    a = ap.parse_args()

    fact_budget = int(C.CTX_MAX_CHARS * a.fact_share)
    pass_budget = C.CTX_MAX_CHARS - fact_budget
    print(f"budget {C.CTX_MAX_CHARS} chars = {pass_budget} passages + {fact_budget} facts "
          f"(identical total to the rag arm)")

    for split in a.splits:
        rag = load(split, "rag")
        onto = load(split, a.onto_arm)
        tag = "hybrid" if abs(a.fact_share - 0.30) < 1e-9 else f"hybrid{int(a.fact_share*100)}"
        out = C.CTX_DIR / f"{split}_{tag}.jsonl"
        n_both = 0
        with open(out, "w", encoding="utf-8") as f:
            for rid, r in rag.items():
                o = onto.get(rid)
                facts = (o or {}).get("context", "")
                if facts.startswith("[no ontology"):
                    facts = ""
                # Facts first: they are short, verifiable, and anchor the entities
                # the passages then illustrate. Passages take the remaining budget.
                fact_block = _truncate(facts, fact_budget) if facts else ""
                pass_block = _truncate(r["context"], pass_budget)
                parts = []
                if fact_block:
                    parts.append("Knowledge-graph facts:\n" + fact_block.split("\n", 1)[-1]
                                 if fact_block.startswith(("Knowledge-graph", "Time-scoped"))
                                 else "Knowledge-graph facts:\n" + fact_block)
                    n_both += 1
                parts.append("Retrieved reference passages:\n" + pass_block)
                ctx = "\n\n".join(parts)
                f.write(json.dumps({
                    "record_id": rid, "Input": r["Input"], "Output": r["Output"],
                    "context": ctx,
                    "n_hits": (o or {}).get("n_hits", 0) + r.get("n_hits", 0),
                    "fact_chars": len(fact_block), "passage_chars": len(pass_block),
                }, ensure_ascii=False) + "\n")
        import statistics as st
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        print(f"  {split}: {len(rows):,} items -> {out.name}  "
              f"ctx {int(st.mean(len(x['context']) for x in rows)):,} chars "
              f"(facts {int(st.mean(x['fact_chars'] for x in rows))}, "
              f"passages {int(st.mean(x['passage_chars'] for x in rows))}), "
              f"{n_both:,} with facts")


if __name__ == "__main__":
    main()
