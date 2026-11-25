"""
WBS Task 3.2.1 - Tuning the ontology-aware retriever's selection policy.

The published V3 context builder picks ~32 entity cards (dense card search plus
intent-to-class routing ordered by graph degree), emits the first 3 facts of
each, then truncates to the shared 3,000-character budget. On the 347-item
benchmark that context carries 38.9% of the gazetteer places the gold answer
names, against 42.3% for the dense-RAG arm.

The ceiling is not the ontology. Every gold gazetteer place is present in the
card corpus as a card NAME, so injecting the whole corpus would score 1.000 and
the 38.9% is a budget-allocation result, not a knowledge gap. This script sweeps
selection policies against that budget - see `tuned_context.POLICIES` for the
ones carried forward, and the exploratory block below for the ones that failed.

Two controls are included deliberately:

  similarity routing   order each routed class by similarity to the question
                       instead of by graph degree. It should be better. It is
                       much worse, which says the gold answers name PROMINENT
                       places, not places that resemble the question.
  global top-degree    no question-dependent retrieval at all: the N
                       most-connected cards, same for every question. If a
                       static roster scores well, the coverage metric is close
                       to degenerate and cannot by itself justify a retriever.

Coverage is a proxy, and the Task 4.1.2 retrieval ablation already measured it as
a weak one (a 29.7-point coverage collapse cost 0.9 points of generated token
F1). So this script only ranks candidates; `generate_tuned_onto` settles it on
real output.

  python -m wbs.task3_retrieval.improve_onto_context --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
OUT = ROOT / "wbs" / "deliverables" / "task3_retrieval"

# policies that are not in the carried-forward table: kept because the negative
# result is the finding
EXPLORATORY = [
    ("rr 1 fact only", dict(n_entities=16, facts_per=1, round_robin=True)),
    ("rr 2 facts", dict(n_entities=16, facts_per=2, round_robin=True)),
    ("rr 1 fact, 32 dense", dict(n_entities=32, facts_per=1, round_robin=True)),
    ("rr 1 fact, 64 dense", dict(n_entities=64, facts_per=1, round_robin=True)),
    ("similarity routing", dict(n_entities=32, facts_per=1, round_robin=True,
                                routing="similarity")),
    ("similarity + hop2 + nohub", dict(n_entities=32, facts_per=1, round_robin=True,
                                       routing="similarity", hop2=8, hub_demote=True)),
    ("hub demote", dict(n_entities=32, facts_per=1, round_robin=True, hub_demote=True)),
    ("routing first, rr 1 fact", dict(n_entities=16, facts_per=1, round_robin=True,
                                      routing_first=True)),
    ("routing only (no dense)", dict(n_entities=0, facts_per=2, round_robin=True,
                                     routing_first=True, per_class=8)),
    ("routing deep 20/class", dict(n_entities=16, facts_per=2, round_robin=True,
                                   routing_first=True, per_class=20)),
    ("validation-clean, dense too", dict(n_entities=16, facts_per=2, round_robin=True,
                                         routing_first=True, per_class=12, hop2=6,
                                         val_drop=True, hub_demote=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()

    from experiments.exp024_rag_vs_ontology import config as C
    from experiments.exp024_rag_vs_ontology.contexts import RagContextBuilder
    from experiments.exp042_prf.compute_prf import ents
    from wbs.task3_retrieval.tuned_context import POLICIES, TunedOntoContext
    from src.utils.proctitle import set_proctitle
    set_proctitle()

    bench = [json.loads(l) for l in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        bench = bench[: a.limit]
    qs = [b["Input"] for b in bench]
    gold = [set(ents(b["Output"])) for b in bench]
    nz = [i for i, g in enumerate(gold) if g]
    print(f"{len(bench)} questions, {len(nz)} with >=1 gold gazetteer place", flush=True)

    onto = TunedOntoContext(device=a.device).prepare(qs)
    print(f"{len(onto.cards):,} entity cards", flush=True)

    def evaluate(name: str, params: dict, carried: bool) -> dict:
        t0 = time.perf_counter()
        ctxs = [onto.build(i, **params) for i in range(len(qs))]
        cov, named = [], []
        for i in nz:
            low = ctxs[i]["context"].lower()
            cov.append(sum(1 for e in gold[i] if e in low) / len(gold[i]))
            named.append(sum(1 for e in gold[i] if e in low))
        row = {"config": name, "carried_forward": carried, "params": params,
               "gold_coverage": st.mean(cov),
               "gold_entities_in_ctx": st.mean(named),
               "ctx_chars": round(st.mean(len(c["context"]) for c in ctxs)),
               "entities_picked": round(st.mean(c["n_entities"] for c in ctxs), 1),
               "facts_emitted": round(st.mean(c["n_facts"] for c in ctxs), 1),
               "build_s": round(time.perf_counter() - t0, 1)}
        print(f"  {'*' if carried else ' '} {name:<30} cov {row['gold_coverage']*100:5.1f}%  "
              f"gold/ctx {row['gold_entities_in_ctx']:4.1f}  "
              f"ents {row['entities_picked']:5.1f}  chars {row['ctx_chars']:,}", flush=True)
        return row

    print("\nnamed policies (* = carried forward to generation):", flush=True)
    rows = [evaluate(k, v, True) for k, v in POLICIES.items()]
    print("\nexploratory, kept for the negative result:", flush=True)
    rows += [evaluate(k, v, False) for k, v in EXPLORATORY]

    rag = RagContextBuilder(device=a.device); rag.load()
    rctx = [r["context"].lower() for r in rag.retrieve_batch(qs)]
    rag_cov = st.mean(sum(1 for e in gold[i] if e in rctx[i]) / len(gold[i]) for i in nz)
    allcards = " || ".join(c["text"].lower() for c in onto.cards)
    ceiling = st.mean(sum(1 for e in gold[i] if e in allcards) / len(gold[i]) for i in nz)

    qdep = [r for r in rows if not r["params"].get("global_degree")]
    best_q = max(qdep, key=lambda r: r["gold_coverage"])
    best_any = max(rows, key=lambda r: r["gold_coverage"])
    base = rows[0]
    out = {"n_questions": len(bench), "n_with_gold": len(nz),
           "mean_gold_entities": round(st.mean(len(gold[i]) for i in nz), 2),
           "budget_chars": C.CTX_MAX_CHARS, "cards": len(onto.cards),
           "corpus_ceiling": ceiling, "rag_reference": rag_cov,
           "baseline": base["config"], "baseline_coverage": base["gold_coverage"],
           "best_question_dependent": best_q["config"],
           "best_question_dependent_coverage": best_q["gold_coverage"],
           "best_any": best_any["config"], "best_any_coverage": best_any["gold_coverage"],
           "results": rows}
    tag = f"__{a.out_tag}" if a.out_tag else ""
    (OUT / f"onto_context_tuning{tag}.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")
    print(f"\nbaseline {base['gold_coverage']*100:.1f}%  ->  best question-dependent "
          f"{best_q['gold_coverage']*100:.1f}% ({best_q['config']})")
    print(f"RAG reference {rag_cov*100:.1f}%  |  static-roster control "
          f"{best_any['gold_coverage']*100:.1f}% ({best_any['config']})  |  "
          f"corpus ceiling {ceiling*100:.1f}%")
    print(f"wrote onto_context_tuning{tag}.json")


if __name__ == "__main__":
    main()
