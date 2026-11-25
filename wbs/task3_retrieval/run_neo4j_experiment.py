"""WBS Task 3.2.1 - the retrieval experiment, run against the Neo4j ontology.

The ontology now lives in Neo4j under the companion project's schema (see
`task3_storage/neo4j_ontology.py`). This runs the offline gold-coverage sweep
with the retrieval arm sourced from that graph, and asks the one question the
file-backed retriever cannot answer for itself:

    a card carries at most 8 facts, so the published 2-hop expansion can only
    follow the hop predicates that survived the cap - 945 of the graph's 3,398.
    Does following the other 72% help?

Three groups are measured in one process, so nothing is compared across runs:

  published      the arms exactly as they run today, card-sourced hop structure
  graph k=2      the same width, hop structure resolved in Cypher over every
                 out-edge. The control: it isolates the hidden structure
  graph k=4,8    wider fan-out, which only the graph can supply

Coverage is a proxy and a weak one - Task 4.1.2 measured a 29.7-point coverage
collapse costing 0.9 points of generated token F1 - so this ranks candidates and
settles nothing on its own.

  CUDA_VISIBLE_DEVICES=5,6 python -m wbs.task3_retrieval.run_neo4j_experiment
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

# The published policies this is measured against, card-sourced hop structure.
PUBLISHED = {
    "baseline": dict(n_entities=16, facts_per=3),
    "routing_deep_hop2": dict(n_entities=16, facts_per=2, round_robin=True,
                              routing_first=True, per_class=12, hop2=6),
    "rr_hop2": dict(n_entities=32, facts_per=1, round_robin=True, hop2=8),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "neo4j_retrieval_sweep.json"))
    a = ap.parse_args()

    from experiments.exp042_prf.compute_prf import ents
    from wbs.task3_retrieval.neo4j_context import GRAPH_POLICIES, Neo4jOntoContext

    bench = [json.loads(line) for line in
             open(EVAL / "tourism_benchmark_categorized.jsonl", encoding="utf-8")
             if line.strip()]
    if a.limit:
        bench = bench[:a.limit]
    qs = [b["Input"] for b in bench]
    gold = [set(ents(b["Output"])) for b in bench]
    nz = [i for i, g in enumerate(gold) if g]
    print(f"{len(bench)} questions, {len(nz)} with >=1 gold gazetteer place",
          flush=True)

    onto = Neo4jOntoContext(device=a.device).prepare(qs)
    gs = onto.graph_stats
    print(f"cards from {onto.cards_source}: {len(onto.cards):,}", flush=True)
    print(f"hop targets: {gs['targets_on_cards']:,} visible on cards, "
          f"{gs['targets_in_graph']:,} in the graph "
          f"({gs['hidden_from_cards']:,} hidden, "
          f"{gs['cards_with_more_targets_in_graph']:,} cards affected)",
          flush=True)

    per_q: dict[str, list[float]] = {}

    def evaluate(name, params, group) -> dict:
        t0 = time.perf_counter()
        ctxs = [onto.build(i, **params) for i in range(len(qs))]
        cov, named = [], []
        for i in nz:
            low = ctxs[i]["context"].lower()
            cov.append(sum(1 for e in gold[i] if e in low) / len(gold[i]))
            named.append(sum(1 for e in gold[i] if e in low))
        per_q[name] = cov
        row = {"config": name, "group": group, "params": params,
               "gold_coverage": st.mean(cov),
               "gold_entities_in_ctx": st.mean(named),
               "ctx_chars": round(st.mean(len(c["context"]) for c in ctxs)),
               "entities_picked": round(st.mean(c["n_entities"] for c in ctxs), 1),
               "facts_emitted": round(st.mean(c["n_facts"] for c in ctxs), 1),
               "build_s": round(time.perf_counter() - t0, 1)}
        print(f"  {name:<24} cov {row['gold_coverage'] * 100:5.1f}%  "
              f"gold/ctx {row['gold_entities_in_ctx']:4.1f}  "
              f"ents {row['entities_picked']:5.1f}  "
              f"chars {row['ctx_chars']:,}", flush=True)
        return row

    print("\npublished policies, hop structure read off the cards:", flush=True)
    rows = [evaluate(k, dict(v, hop_from="cards"), "published")
            for k, v in PUBLISHED.items()]

    print("\nhop structure resolved in Neo4j over every out-edge:", flush=True)
    rows += [evaluate(k, dict(v, hop_from="graph"), "graph")
             for k, v in GRAPH_POLICIES.items()]

    def paired(a_name: str, b_name: str, iters: int = 10000) -> dict:
        """Is b better than a, on the same questions?

        The coverage figures are deterministic, so there is no run-to-run noise
        to average away - the only question is whether the per-question
        differences are consistent or a handful of questions carrying the mean.
        A paired bootstrap over questions answers that; the sign test says how
        many questions moved at all.
        """
        import random
        A, B = per_q.get(a_name), per_q.get(b_name)
        if not A or not B or len(A) != len(B):
            return {}
        d = [b - a for a, b in zip(A, B)]
        n = len(d)
        better = sum(1 for x in d if x > 1e-12)
        worse = sum(1 for x in d if x < -1e-12)
        rng = random.Random(7)
        means = []
        for _ in range(iters):
            means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        return {
            "a": a_name, "b": b_name, "n_questions": n,
            "mean_difference": sum(d) / n,
            "ci95": [means[int(0.025 * iters)], means[int(0.975 * iters)]],
            "questions_better": better, "questions_worse": worse,
            "questions_unchanged": n - better - worse,
        }

    pub = {r["config"]: r for r in rows if r["group"] == "published"}
    graph = {r["config"]: r for r in rows if r["group"] == "graph"}
    control = graph.get("graph_hop2_k2")
    anchor = pub.get("routing_deep_hop2")

    # The control has the published width and the graph's hop structure, so the
    # gap between it and the published arm is the hidden structure on its own.
    hidden_effect = (None if not (control and anchor) else
                     control["gold_coverage"] - anchor["gold_coverage"])
    best = max(rows, key=lambda r: r["gold_coverage"])
    best_graph = max(graph.values(), key=lambda r: r["gold_coverage"]) \
        if graph else None

    out = {
        "ontology_source": "neo4j",
        "cards_source": onto.cards_source,
        "graph": gs,
        "questions": len(bench), "questions_with_gold": len(nz),
        "rows": rows,
        "published_anchor": anchor["config"] if anchor else None,
        "published_anchor_coverage": anchor["gold_coverage"] if anchor else None,
        "same_width_graph_hop_coverage": control["gold_coverage"] if control else None,
        "hidden_structure_effect": hidden_effect,
        "best": best["config"], "best_coverage": best["gold_coverage"],
        "best_graph_policy": best_graph["config"] if best_graph else None,
        "best_graph_coverage": best_graph["gold_coverage"] if best_graph else None,
        "paired_control": paired("routing_deep_hop2", "graph_hop2_k2"),
        "paired_best": paired("routing_deep_hop2",
                              best_graph["config"] if best_graph else ""),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    if hidden_effect is not None:
        pts = hidden_effect * 100
        print(f"same policy, same width, hop structure from the graph instead of "
              f"the cards: {pts:+.2f} points of coverage")
        if abs(pts) < 0.5:
            print("  i.e. nothing. The 72% of hop structure the cards hide does "
                  "not reach the context at this fan-out.")
    if best_graph and anchor:
        print(f"widest graph policy ({best_graph['config']}): "
              f"{best_graph['gold_coverage'] * 100:.1f}% vs "
              f"{anchor['gold_coverage'] * 100:.1f}% published "
              f"({(best_graph['gold_coverage'] - anchor['gold_coverage']) * 100:+.2f})")
    pb = out["paired_best"]
    if pb:
        lo, hi = pb["ci95"]
        print(f"  paired over {pb['n_questions']} questions: "
              f"{pb['mean_difference'] * 100:+.2f} points, "
              f"95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]")
        print(f"  {pb['questions_better']} questions better, "
              f"{pb['questions_worse']} worse, "
              f"{pb['questions_unchanged']} unchanged")
        if lo <= 0 <= hi:
            print("  the interval covers zero: not separable on this metric")
    print("\nCoverage is a weak proxy. Task 4.1.2 measured a 29.7-point "
          "coverage collapse costing 0.9 points of generated token F1, so a "
          "gain of about a point here is worth roughly 0.03 F1 - below what "
          "generation would resolve. Treat this as a ranking, not a result.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
