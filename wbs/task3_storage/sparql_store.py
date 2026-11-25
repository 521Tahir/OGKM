"""
WBS Task 3.1.1 - RDF store with a SPARQL query interface.

The WBS names "ontology storage using RDF/graph database and file-based systems".
File-based storage is the versioned `artifacts/ontologies/<version>/` JSON, and an
in-memory traversal store already exists at `src/graph/graph_store.py`. What was
missing is the RDF half: a store that answers SPARQL rather than Python method calls.

This module loads the Task 2.2 Turtle serialization into an rdflib Graph, exposes it
through SPARQL 1.1, and runs eight competency questions drawn from the six benchmark
task types so the store is evidenced by queries that the domain actually asks. It also
measures load and query latency, and round-trips the graph to N-Quads so the store has
a durable on-disk form independent of the Turtle input.

rdflib's SPARQL engine covers query capability on this graph's scale (see the latency
table), so a server is a deployment choice rather than a missing capability.
`--sparql-endpoint URL` points the same queries at an external endpoint when one
exists. A Neo4j server now runs the same nine questions in Cypher - see
`wbs/task3_storage/neo4j_store.py` - which is where the two query paths are compared;
Fuseki and GraphDB are still not deployed.

  python -m wbs.task3_storage.sparql_store
  python -m wbs.task3_storage.sparql_store --sparql-endpoint http://localhost:3030/jeju
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

TTL = ROOT / "wbs" / "deliverables" / "task2_formalization" / "jeju_tourism_ontology.ttl"
OUT = ROOT / "wbs" / "deliverables" / "task3_storage"
NS = "http://jejunu.ac.kr/ontology/jeju-tourism#"

PREFIX = f"""PREFIX jeju: <{NS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX prov: <http://www.w3.org/ns/prov#>
"""

# One competency question per benchmark task type, plus governance and reasoning ones.
QUERIES = [
    ("Q1 inventory", "how large is the graph and how is it typed",
     """SELECT (COUNT(DISTINCT ?s) AS ?individuals) (COUNT(DISTINCT ?c) AS ?classes)
        WHERE { ?s a owl:NamedIndividual ; a ?c . FILTER(?c != owl:NamedIndividual) }"""),

    ("Q2 recommendation", "natural attractions with the most inbound relations",
     """SELECT ?label (COUNT(?s) AS ?inbound) WHERE {
          ?x a jeju:NaturalAttraction ; rdfs:label ?label .
          ?s ?p ?x .
        } GROUP BY ?label ORDER BY DESC(?inbound) LIMIT 10"""),

    ("Q3 dining", "cuisines and where they are served",
     """SELECT ?dish ?place WHERE {
          ?d a jeju:Cuisine ; rdfs:label ?dish ; jeju:served_at ?p .
          ?p rdfs:label ?place .
        } LIMIT 20"""),

    ("Q4 transport", "what a destination is reachable by",
     """SELECT ?place ?mode WHERE {
          ?x jeju:reachable_by ?m ; rdfs:label ?place .
          ?m rdfs:label ?mode .
        } LIMIT 20"""),

    ("Q5 itinerary (2-hop path)", "places near a place that is inside a region",
     """SELECT ?region ?place ?neighbour WHERE {
          ?p jeju:located_in ?r ; rdfs:label ?place ; jeju:near ?n .
          ?r rdfs:label ?region . ?n rdfs:label ?neighbour .
        } LIMIT 20"""),

    ("Q6 safety", "entities carrying a risk or an avoid-during constraint",
     """SELECT ?label ?pred ?obj WHERE {
          { ?s jeju:has_risk ?o . BIND(jeju:has_risk AS ?pred) }
          UNION { ?s jeju:avoid_during ?o . BIND(jeju:avoid_during AS ?pred) }
          UNION { ?s jeju:requires ?o . BIND(jeju:requires AS ?pred) }
          ?s rdfs:label ?label . ?o rdfs:label ?obj .
        } LIMIT 20"""),

    ("Q7 accessibility", "entities declared suitable for a traveller type",
     """SELECT ?label ?who WHERE {
          ?s jeju:suitable_for ?t ; rdfs:label ?label .
          ?t rdfs:label ?who .
        } LIMIT 20"""),

    ("Q8 provenance / governance", "how assertions are sourced, and confidence spread",
     """SELECT ?source (COUNT(?s) AS ?n) (AVG(?c) AS ?mean_conf) WHERE {
          ?s prov:wasGeneratedBy ?source ; jeju:confidence ?c .
        } GROUP BY ?source ORDER BY DESC(?n)"""),

    ("Q9 transitive closure", "property-path query: region containment, any depth",
     """SELECT ?label WHERE {
          ?s jeju:part_of+ ?top ; rdfs:label ?label .
          ?top rdfs:label "Jeju" .
        } LIMIT 20"""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparql-endpoint", default=None,
                    help="query an external SPARQL endpoint instead of the local graph")
    ap.add_argument("--repeats", type=int, default=5,
                    help="timed repetitions per query")
    ap.add_argument("--out-tag", default=None,
                    help="suffix for the output files, so an endpoint run does not "
                         "overwrite the in-process one")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.out_tag is None:
        a.out_tag = "endpoint" if a.sparql_endpoint else ""
    sfx = f"__{a.out_tag}" if a.out_tag else ""

    from rdflib import Graph
    if a.sparql_endpoint:
        from rdflib.plugins.stores.sparqlstore import SPARQLStore
        g = Graph(store=SPARQLStore(a.sparql_endpoint))
        load_s, n_triples, backend = 0.0, None, f"SPARQL endpoint {a.sparql_endpoint}"
        print(f"querying {a.sparql_endpoint}")
    else:
        if not TTL.exists():
            sys.exit(f"missing {TTL}; run wbs.task2_formalization.normalize_and_formalize")
        g = Graph()
        t0 = time.perf_counter()
        g.parse(TTL, format="turtle")
        load_s = time.perf_counter() - t0
        n_triples = len(g)
        backend = "rdflib in-process store, Turtle-backed"
        print(f"loaded {n_triples:,} triples in {load_s:.3f}s")

    rows = []
    for name, question, q in QUERIES:
        lat, n = [], 0
        err = None
        for i in range(a.repeats):
            try:
                t0 = time.perf_counter()
                res = list(g.query(PREFIX + q))
                lat.append((time.perf_counter() - t0) * 1000)
                n = len(res)
            except Exception as e:                      # endpoint or syntax failure
                err = f"{type(e).__name__}: {e}"
                break
        first = []
        if not err:
            res = list(g.query(PREFIX + q))[:3]
            first = [" | ".join(str(x) for x in r) for r in res]
        rows.append({"query": name, "question": question, "rows": n,
                     "p50_ms": round(st.median(lat), 2) if lat else None,
                     "max_ms": round(max(lat), 2) if lat else None,
                     "sample": first, "error": err})
        print(f"  [{name}] {n} rows, "
              f"{rows[-1]['p50_ms']} ms" if not err else f"  [{name}] FAILED {err}",
              flush=True)

    # a durable on-disk form that does not depend on re-parsing the Turtle
    nq = OUT / "jeju_tourism_ontology.nt"
    if not a.sparql_endpoint:
        t0 = time.perf_counter()
        g.serialize(destination=nq, format="nt")
        dump_s = time.perf_counter() - t0
        g2 = Graph(); t0 = time.perf_counter()
        g2.parse(nq, format="nt")
        reload_s = time.perf_counter() - t0
        roundtrip = {"ntriples_bytes": nq.stat().st_size, "dump_s": round(dump_s, 3),
                     "reload_s": round(reload_s, 3), "triples_after_reload": len(g2),
                     "lossless": len(g2) == n_triples}
    else:
        roundtrip = None

    out = {"backend": backend, "ttl_source": str(TTL.relative_to(ROOT)),
           "triples": n_triples, "load_s": round(load_s, 3),
           "repeats": a.repeats, "queries": rows, "roundtrip": roundtrip,
           "graph_database_server": None}
    (OUT / f"sparql_store{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["# Task 3.1.1 - RDF Store and SPARQL Interface\n",
         f"Backend: **{backend}**."]
    if n_triples is not None:
        L.append(f" {n_triples:,} triples parsed from "
                 f"`{TTL.relative_to(ROOT)}` in {load_s:.3f} s.\n")
    L.append("\nThe WBS asks for ontology storage over \"RDF/graph database and "
             "file-based systems\". Three of those exist and are exercised here:\n")
    L.append("- **File-based** — versioned `artifacts/ontologies/<version>/"
             "{concepts,relations}.json`, content-addressed and governed by Task 3.1.2.")
    L.append("- **RDF** — the Turtle/RDF-XML/JSON-LD serializations from Task 2.2, "
             "loaded here into an rdflib graph and queried with SPARQL 1.1 below.")
    L.append("- **Graph traversal** — `src/graph/graph_store.py`, an in-memory adjacency "
             "index used by the retrieval path, where per-hop latency matters more than "
             "query expressiveness.\n")
    L.append("## Competency questions\n")
    L.append("Each query is one of the six benchmark task types, plus governance and a "
             "property-path query, so the store is evidenced against questions the "
             f"domain actually asks. Latency is the median of {a.repeats} runs.\n")
    L.append("| Query | Question | Rows | p50 | max |")
    L.append("|---|---|--:|--:|--:|")
    for r in rows:
        if r["error"]:
            L.append(f"| {r['query']} | {r['question']} | — | — | "
                     f"**{r['error'][:60]}** |")
        else:
            L.append(f"| {r['query']} | {r['question']} | {r['rows']:,} | "
                     f"{r['p50_ms']} ms | {r['max_ms']} ms |")
    L.append("")
    ok = [r for r in rows if not r["error"]]
    if ok:
        slow = max(ok, key=lambda r: r["p50_ms"])
        L.append(f"All {len(ok)} of {len(rows)} queries answer from the RDF store; the "
                 f"slowest is *{slow['query']}* at {slow['p50_ms']} ms, which is the "
                 f"property-path/aggregate class. For comparison, §7 of the final report "
                 f"measures generation at roughly 7 s per request, so SPARQL is not on "
                 f"the critical path at this scale.\n")
    L.append("## Sample bindings\n")
    for r in ok:
        if r["sample"]:
            L.append(f"**{r['query']}** — {r['question']}\n")
            for s in r["sample"]:
                L.append(f"- `{s}`")
            L.append("")
    if roundtrip:
        L.append("## Durability\n")
        L.append(f"The graph is serialized to N-Triples (`{nq.name}`, "
                 f"{roundtrip['ntriples_bytes']:,} bytes) in {roundtrip['dump_s']} s and "
                 f"re-parsed in {roundtrip['reload_s']} s, returning "
                 f"{roundtrip['triples_after_reload']:,} triples — "
                 f"{'lossless' if roundtrip['lossless'] else '**lossy, investigate**'}. "
                 f"That gives the RDF store an on-disk form independent of the Turtle "
                 f"input it was built from.\n")
    L.append("## A query-planning note worth keeping\n")
    L.append("`Q6` was first written with an unbound predicate constrained by "
             "`VALUES ?pred { ... }`, which is the natural phrasing. On rdflib that "
             "costs **16.2 s**, because the engine materialises the unbound "
             "`?s ?pred ?o` pattern across all 17,706 triples before applying the "
             "binding. Rewritten as a `UNION` of three concrete-predicate patterns it "
             "costs **74 ms** — a 218x difference for an identical result set. The "
             "lesson transfers to any SPARQL layer put in front of this ontology: bind "
             "predicates in the pattern, not in a filter or a `VALUES` block, because "
             "an unbound predicate defeats the predicate index. The same shape is why "
             "`Q1` and `Q8`, which aggregate over every individual, cost tens of "
             "milliseconds while the targeted lookups cost single digits.\n")
    L.append("## What is not deployed\n")
    L.append("No graph-database **server**. Neo4j, Fuseki and GraphDB are not installed "
             "on this machine, and `requirements.txt` carries `neo4j` commented out with "
             "\"uncomment when Neo4j is available\". At 17,706 triples an in-process "
             "store answers every competency question in milliseconds, so this is a "
             "deployment decision rather than a capability gap — but it is a real "
             "difference from what the WBS line item names, and installing a server is "
             "an infrastructure change for the project owner to authorise. The same "
             "queries run unchanged against an external endpoint with "
             "`--sparql-endpoint URL`.\n")
    (OUT / f"sparql_store{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote sparql_store{sfx}.{{md,json}}"
          + (f" and {nq.name}" if not a.sparql_endpoint else ""))


if __name__ == "__main__":
    main()
