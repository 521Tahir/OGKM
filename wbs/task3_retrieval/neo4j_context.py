"""WBS Task 3.2.1 - the retrieval arm, sourced from the Neo4j ontology.

`tuned_context.TunedOntoContext` builds its entity cards by reading
`artifacts/ontologies/<version>/` off disk. This module builds the same cards
from the Neo4j graph loaded by `neo4j_storage.neo4j_ontology`, which holds the
ontology under the companion project's schema:

    (:Concept {name, concept_class, confidence, aliases, source})
    (:Concept)-[:RELATES_TO {predicate, confidence, source, ordinal}]->(:Concept)

Two things follow from that, and only the second is interesting.

1. Parity. The cards the graph produces are identical to the cards the files
   produce - same names, classes, degrees, facts and fact order - so swapping the
   store cannot move a single published number. `card_parity()` checks it rather
   than asserting it, and the experiment refuses to run if it fails.

2. Headroom. A card keeps at most 8 facts, so the card-based 2-hop expansion can
   only follow the hop predicates that survived that cap. Measured on v1.4.0:
   the cards expose a fraction of the hop targets the graph holds, so most of
   the hop structure is invisible to a retriever that reads cards. The published
   policy then follows only the first 2 targets of each hopped-from card, so the
   hidden structure is large and the policy that would use it is narrow - which
   is why this is worth measuring rather than assuming.

Note the version. The storage, reporting and figure deliverables are all on
v1.4.0; this experiment runs on v1.2.0 (1,585 concepts / 9,208 relations), which
is what `exp024_rag_vs_ontology/config.py` pins and what the FAISS card index was
built against. The graph has to hold the version the experiment runs on, and
Neo4j Community serves one database, so the two cannot be resident at once.

The graph policies below resolve hop targets with Cypher over every out-edge and
widen the fan-out, which is the thing the file-backed retriever cannot do.

  python -m wbs.task3_retrieval.neo4j_context --check   # parity only, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wbs.task3_retrieval.tuned_context import HOP_PREDS, TunedOntoContext  # noqa: E402

# The predicates as the ontology spells them; `tuned_context` matches them in
# prose ("located in") because it reads them back out of rendered fact strings.
HOP_PREDICATES = tuple(p.replace(" ", "_") for p in HOP_PREDS)

HOP_TARGETS_CYPHER = """
MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
WHERE r.predicate IN $preds AND toLower(a.name) <> toLower(b.name)
WITH a, r, b ORDER BY r.ordinal
RETURN a.name AS name, collect(b.name) AS targets
"""

# Policies that spend the graph's extra hop structure. `graph_hop2_k2` is the
# control: the same width as the published policy, resolved in the graph instead
# of off the cards, so any difference between it and `routing_deep_hop2` is the
# hidden structure and nothing else.
GRAPH_POLICIES: dict[str, dict] = {
    "graph_hop2_k2": dict(n_entities=16, facts_per=2, round_robin=True,
                          routing_first=True, per_class=12, hop2=6, fanout=2),
    "graph_hop2_k4": dict(n_entities=16, facts_per=2, round_robin=True,
                          routing_first=True, per_class=12, hop2=6, fanout=4),
    "graph_hop2_k8": dict(n_entities=16, facts_per=2, round_robin=True,
                          routing_first=True, per_class=12, hop2=6, fanout=8),
    "graph_hop_wide": dict(n_entities=16, facts_per=2, round_robin=True,
                           routing_first=True, per_class=12, hop2=12, fanout=4),
    "graph_rr_hop2_k4": dict(n_entities=32, facts_per=1, round_robin=True,
                             hop2=8, fanout=4),
}


class Neo4jOntoContext(TunedOntoContext):
    """TunedOntoContext with the cards and the hop graph read from Neo4j."""

    def __init__(self, device: str = "cuda", cards_from_graph: bool = True):
        super().__init__(device=device)
        self.cards_from_graph = cards_from_graph
        self.graph_stats: dict = {}

    def load(self):
        if self._ready:
            return self
        super().load()

        from wbs.task3_storage.neo4j_ontology import OntologyGraph
        g = OntologyGraph()
        try:
            if self.cards_from_graph:
                graph_cards = g.cards()
                report = card_parity(self.cards, graph_cards)
                if not report["identical"]:
                    from experiments.exp024_rag_vs_ontology import config as _C
                    raise RuntimeError(
                        "the cards Neo4j produces differ from the cards the "
                        "experiment builds from files. The usual cause is that "
                        f"the graph holds a different ontology version: the "
                        f"experiment runs on v{_C.ONTO_VERSION}, so load that "
                        f"one with\n  python -m wbs.task3_storage."
                        f"neo4j_ontology --version {_C.ONTO_VERSION} --clear\n"
                        f"{report}")
                # Same content, so the substitution is a provenance change, not
                # a behaviour change. Keep the file order: the FAISS index was
                # built against it and the card index is what `didx` returns.
                self.cards_source = "neo4j"
            else:
                self.cards_source = "files"

            by_name = {c["name"].lower(): i for i, c in enumerate(self.cards)}
            rows = g._read(HOP_TARGETS_CYPHER, preds=list(HOP_PREDICATES))
        finally:
            g.close()

        card_hop = sum(len(t) for t in self.hop_targets)
        graph_targets = [[] for _ in self.cards]
        total = 0
        for r in rows:
            i = by_name.get(str(r["name"]).lower())
            if i is None:
                continue
            seen = set()
            for name in r["targets"]:
                j = by_name.get(str(name).lower())
                if j is None or j in seen:
                    continue
                seen.add(j)
                graph_targets[i].append(j)
            total += len(graph_targets[i])

        self.card_hop_targets = self.hop_targets
        self.graph_hop_targets = graph_targets
        self.hop_targets = graph_targets
        self.graph_stats = {
            "hop_predicates": list(HOP_PREDICATES),
            "targets_on_cards": card_hop,
            "targets_in_graph": total,
            "hidden_from_cards": total - card_hop,
            "cards_with_more_targets_in_graph": sum(
                1 for a, b in zip(self.card_hop_targets, graph_targets)
                if len(b) > len(a)),
            "cards": len(self.cards),
        }
        return self

    def build(self, qi: int, fanout: int = 2, hop_from: str = "graph",
              **kw) -> dict:
        """`fanout` is how many hop targets to follow per hopped-from card.

        `hop_from` selects which hop structure to follow: "graph" is every
        out-edge in Neo4j, "cards" is only the ones the 8-fact cap left visible,
        which is what the published arms ran. Both are needed in one process, or
        the comparison between them would also be a comparison between two runs.
        """
        self.hop_fanout = int(fanout)
        self.hop_targets = (self.card_hop_targets if hop_from == "cards"
                            else self.graph_hop_targets)
        try:
            return super().build(qi, **kw)
        finally:
            self.hop_fanout = 2


def card_parity(file_cards: list[dict], graph_cards: list[dict]) -> dict:
    """Are the two card sets the same ontology, down to fact order?"""
    F = {c["name"]: c for c in file_cards}
    G = {c["name"]: c for c in graph_cards}
    shared = set(F) & set(G)
    worse = lambda key: sorted(n for n in shared if F[n][key] != G[n][key])
    cls, deg, txt = worse("klass"), worse("degree"), worse("text")
    return {
        "file_cards": len(file_cards), "graph_cards": len(graph_cards),
        "only_in_files": sorted(set(F) - set(G))[:10],
        "only_in_graph": sorted(set(G) - set(F))[:10],
        "different_class": cls[:10], "different_degree": deg[:10],
        "different_text": txt[:10],
        "identical": (set(F) == set(G) and not cls and not deg and not txt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="card parity and hop headroom only; no GPU, no model")
    a = ap.parse_args()

    if not a.check:
        print("nothing to do without --check; the sweep lives in "
              "wbs.task3_retrieval.run_neo4j_experiment")
        return

    import json as _json
    from collections import defaultdict
    from wbs.task3_storage.neo4j_ontology import OntologyGraph

    # Rebuild the file cards without importing the GPU context builder.
    from experiments.exp024_rag_vs_ontology import config as C
    print(f"ontology version the experiment runs on: v{C.ONTO_VERSION}")
    base = ROOT / "artifacts" / "ontologies" / C.ONTO_VERSION
    rels = _json.loads((base / "relations.json").read_text(encoding="utf-8"))
    cons = _json.loads((base / "concepts.json").read_text(encoding="utf-8"))
    cls_of = {c.get("name"): c.get("concept_class") for c in cons if c.get("name")}
    out_edges, degree = defaultdict(list), defaultdict(int)
    for r in rels:
        s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), \
            str(r.get("object", ""))
        if not s or not o or s.lower() == o.lower():
            continue
        out_edges[s].append((p, o))
        degree[s] += 1
        degree[o] += 1
    file_cards = []
    for name, edges in out_edges.items():
        if len(name) < 3:
            continue
        klass = cls_of.get(name) or "Concept"
        seen, facts = set(), []
        for p, o in edges:
            k = (p.lower(), o.lower())
            if k in seen:
                continue
            seen.add(k)
            facts.append(f"{name} {p.replace('_', ' ').strip()} {o}")
            if len(facts) >= 8:
                break
        file_cards.append({"name": name, "klass": klass, "degree": degree[name],
                           "facts": facts,
                           "text": f"{name} ({klass}). " + "; ".join(facts)})

    g = OntologyGraph()
    try:
        graph_cards = g.cards()
        rep = card_parity(file_cards, graph_cards)
        rows = g._read(HOP_TARGETS_CYPHER, preds=list(HOP_PREDICATES))
    finally:
        g.close()

    print(f"cards: {rep['file_cards']:,} from files, {rep['graph_cards']:,} "
          f"from Neo4j")
    print(f"identical (names, classes, degrees, fact order): {rep['identical']}")
    for k in ("only_in_files", "only_in_graph", "different_class",
              "different_degree", "different_text"):
        if rep[k]:
            print(f"  {k}: {rep[k]}")

    names = {c["name"].lower(): c for c in file_cards}
    card_t = 0
    for c in file_cards:
        for f in c["facts"]:
            low = f.lower()
            for p in HOP_PREDS:
                k = low.find(f" {p} ")
                if k >= 0:
                    if f[k + len(p) + 2:].strip().lower() in names:
                        card_t += 1
                    break
    graph_t = sum(len({t.lower() for t in r["targets"]
                       if t.lower() in names}) for r in rows)
    print(f"\nhop targets on the 8-fact cards : {card_t:,}")
    print(f"hop targets in the graph        : {graph_t:,}")
    print(f"invisible to a card-based hop   : {graph_t - card_t:,} "
          f"({(graph_t - card_t) / max(graph_t, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
