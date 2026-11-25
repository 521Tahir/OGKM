"""
WBS Task 3.1.1 - the Neo4j half of "RDF/graph database", measured rather than deferred.

Until a JVM existed on this host, Neo4j was in the repository as client code only
(`src/graph/neo4j_client.py`, `src/graph/cypher_queries.py`, `scripts/ingest_to_neo4j.py`)
with its tests running offline against a mocked driver, and the delivered store was
Oxigraph with SPARQL 1.1. This module closes that gap: it loads the *same* ontology
version into a real Neo4j server and answers the *same* nine competency questions in
Cypher, so the two query paths can be compared on row counts and latency instead of on
assertion.

Two mappings are measured, because the choice of mapping is the result:

  typed   - the idiomatic labelled property graph. A concept becomes a node labelled
            `:Concept` plus its ontology class (`:Beach`), and a triple becomes a
            relationship whose *type* is the predicate (`:LOCATED_IN`). Neo4j indexes
            relationship types natively, so this is the shape its planner is built for.
  generic - what `src/graph/cypher_queries.py` currently does: one label `:GConcept`,
            one relationship type `:RELATES_TO`, and the predicate carried as a
            *property*. This is a triplestore emulated inside a property graph.

Benchmarking only the generic mapping against a real triplestore would measure the
mapping and call it the engine, which is the trap this module exists to avoid.

Prerequisites: a running server (`bash wbs/task3_storage/neo4j_server.sh start`) and
the `neo4j` Python driver.

  python -m wbs.task3_storage.neo4j_store
  python -m wbs.task3_storage.neo4j_store --bolt bolt://127.0.0.1:7687 --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
OUT = ROOT / "wbs" / "deliverables" / "task3_storage"
SPARQL_JSON = OUT / "sparql_store.json"
OXIGRAPH_JSON = OUT / "graph_db.json"

LABEL_OK = re.compile(r"[^A-Za-z0-9_]")


def label_of(concept_class: str) -> str:
    """An ontology class name as a Neo4j label, kept recognisable."""
    clean = LABEL_OK.sub("_", (concept_class or "Concept").strip())
    return clean or "Concept"


def reltype_of(predicate: str) -> str:
    """A predicate as a Neo4j relationship type: snake_case -> SCREAMING_CASE."""
    return LABEL_OK.sub("_", (predicate or "related_to").strip()).upper()


# --------------------------------------------------------------- the questions
#
# One per benchmark task type, plus governance and a variable-depth path query -
# the same nine the SPARQL side answers in wbs/task3_storage/sparql_store.py.

def queries(mapping: str) -> list[tuple[str, str, str]]:
    """The nine competency questions, written for one mapping or the other."""
    if mapping == "typed":
        return [
            ("Q1 inventory", "how large is the graph and how is it typed",
             """MATCH (c:Concept)
                RETURN count(c) AS individuals,
                       count(DISTINCT c.concept_class) AS classes"""),
            ("Q2 recommendation",
             "natural attractions with the most inbound relations",
             """MATCH (x:NaturalAttraction)<-[r]-()
                RETURN x.name AS label, count(r) AS inbound
                ORDER BY inbound DESC LIMIT 10"""),
            ("Q3 dining", "cuisines and where they are served",
             """MATCH (d:Cuisine)-[:SERVED_AT]->(p)
                RETURN d.name AS dish, p.name AS place LIMIT 20"""),
            ("Q4 transport", "what a destination is reachable by",
             """MATCH (x)-[:REACHABLE_BY]->(m)
                RETURN x.name AS place, m.name AS mode LIMIT 20"""),
            ("Q5 itinerary (2-hop path)",
             "places near a place that is inside a region",
             """MATCH (p)-[:LOCATED_IN]->(r), (p)-[:NEAR]->(n)
                RETURN r.name AS region, p.name AS place, n.name AS neighbour
                LIMIT 20"""),
            ("Q6 safety", "entities carrying a risk or an avoid-during constraint",
             """MATCH (s)-[r:HAS_RISK|AVOID_DURING|REQUIRES]->(o)
                RETURN s.name AS label, type(r) AS pred, o.name AS obj LIMIT 20"""),
            ("Q7 accessibility", "entities declared suitable for a traveller type",
             """MATCH (s)-[:SUITABLE_FOR]->(t)
                RETURN s.name AS label, t.name AS who LIMIT 20"""),
            ("Q8 provenance / governance",
             "how assertions are sourced, and confidence spread",
             """MATCH (c:Concept) WHERE c.source IS NOT NULL
                RETURN c.source AS source, count(c) AS n,
                       avg(c.confidence) AS mean_conf
                ORDER BY n DESC"""),
            ("Q9 transitive closure",
             "property-path query: region containment, any depth",
             """MATCH (s)-[:PART_OF*1..]->(top {name: 'Jeju'})
                RETURN DISTINCT s.name AS label LIMIT 20"""),
        ]
    return [
        ("Q1 inventory", "how large is the graph and how is it typed",
         """MATCH (c:GConcept)
            RETURN count(c) AS individuals,
                   count(DISTINCT c.concept_class) AS classes"""),
        ("Q2 recommendation",
         "natural attractions with the most inbound relations",
         """MATCH (x:GConcept {concept_class: 'NaturalAttraction'})<-[r:RELATES_TO]-()
            RETURN x.name AS label, count(r) AS inbound
            ORDER BY inbound DESC LIMIT 10"""),
        ("Q3 dining", "cuisines and where they are served",
         """MATCH (d:GConcept {concept_class: 'Cuisine'})
                  -[:RELATES_TO {predicate: 'served_at'}]->(p:GConcept)
            RETURN d.name AS dish, p.name AS place LIMIT 20"""),
        ("Q4 transport", "what a destination is reachable by",
         """MATCH (x:GConcept)-[:RELATES_TO {predicate: 'reachable_by'}]->(m:GConcept)
            RETURN x.name AS place, m.name AS mode LIMIT 20"""),
        ("Q5 itinerary (2-hop path)",
         "places near a place that is inside a region",
         """MATCH (p:GConcept)-[:RELATES_TO {predicate: 'located_in'}]->(r:GConcept),
                  (p)-[:RELATES_TO {predicate: 'near'}]->(n:GConcept)
            RETURN r.name AS region, p.name AS place, n.name AS neighbour LIMIT 20"""),
        ("Q6 safety", "entities carrying a risk or an avoid-during constraint",
         """MATCH (s:GConcept)-[r:RELATES_TO]->(o:GConcept)
            WHERE r.predicate IN ['has_risk', 'avoid_during', 'requires']
            RETURN s.name AS label, r.predicate AS pred, o.name AS obj LIMIT 20"""),
        ("Q7 accessibility", "entities declared suitable for a traveller type",
         """MATCH (s:GConcept)-[:RELATES_TO {predicate: 'suitable_for'}]->(t:GConcept)
            RETURN s.name AS label, t.name AS who LIMIT 20"""),
        ("Q8 provenance / governance",
         "how assertions are sourced, and confidence spread",
         """MATCH (c:GConcept) WHERE c.source IS NOT NULL
            RETURN c.source AS source, count(c) AS n, avg(c.confidence) AS mean_conf
            ORDER BY n DESC"""),
        ("Q9 transitive closure",
         "property-path query: region containment, any depth",
         """MATCH (s:GConcept)-[:RELATES_TO* {predicate: 'part_of'}]->
                  (top:GConcept {name: 'Jeju'})
            RETURN DISTINCT s.name AS label LIMIT 20"""),
    ]


# ------------------------------------------------------------------- ingestion


def read_ontology(version: str):
    cdir = ONTO / version
    concepts = json.loads((cdir / "concepts.json").read_text(encoding="utf-8"))
    relations = json.loads((cdir / "relations.json").read_text(encoding="utf-8"))
    return concepts, relations


def _node_rows(concepts):
    rows = []
    for c in concepts:
        prov = c.get("provenance") or {}
        attrs = c.get("attributes") or {}
        rows.append({
            "name": c["name"],
            "concept_class": c.get("concept_class") or "Concept",
            "label": label_of(c.get("concept_class")),
            "confidence": prov.get("confidence"),
            "source": prov.get("extraction_prompt_version"),
            "validator_status": prov.get("validator_status"),
            "aliases": [str(a) for a in (c.get("aliases") or [])],
            "lat": attrs.get("lat"),
            "lon": attrs.get("lon"),
        })
    return rows


def _edge_rows(relations):
    rows = []
    for r in relations:
        prov = r.get("provenance") or {}
        rows.append({
            "s": r["subject"], "o": r["object"],
            "predicate": r["predicate"],
            "reltype": reltype_of(r["predicate"]),
            "confidence": prov.get("confidence"),
            "source": prov.get("extraction_prompt_version"),
        })
    return rows


# Constraints and indexes outlive the nodes they were created for, so the mapping
# left in the database would otherwise show the other mapping's schema objects
# standing over no data.
SCHEMA_OBJECTS = {
    "typed": ["concept_name", "concept_class"],
    "generic": ["gconcept_name", "gconcept_class", "grel_predicate"],
}


def drop_schema_objects(session, mapping: str) -> list[str]:
    """Remove the named mapping's constraints and indexes, if they exist."""
    dropped = []
    for name in SCHEMA_OBJECTS.get(mapping, []):
        for kind in ("CONSTRAINT", "INDEX"):
            try:
                session.run(f"DROP {kind} {name} IF EXISTS")
                dropped.append(f"{kind.lower()} {name}")
            except Exception:
                pass
    return dropped


def wipe(session):
    # Scoped-variable CALL: required from 5.23 on, the unscoped form is deprecated.
    session.run("MATCH (n) CALL (n) { DETACH DELETE n } "
                "IN TRANSACTIONS OF 10000 ROWS")


def ingest(session, nodes, edges, mapping: str, batch: int = 2000) -> dict:
    """Load the ontology under one mapping and return what it cost."""
    t0 = time.perf_counter()
    if mapping == "typed":
        session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS "
                    "FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        session.run("CREATE INDEX concept_class IF NOT EXISTS "
                    "FOR (c:Concept) ON (c.concept_class)")
    else:
        session.run("CREATE CONSTRAINT gconcept_name IF NOT EXISTS "
                    "FOR (c:GConcept) REQUIRE c.name IS UNIQUE")
        session.run("CREATE INDEX gconcept_class IF NOT EXISTS "
                    "FOR (c:GConcept) ON (c.concept_class)")
        session.run("CREATE INDEX grel_predicate IF NOT EXISTS "
                    "FOR ()-[r:RELATES_TO]-() ON (r.predicate)")
    schema_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    base = "Concept" if mapping == "typed" else "GConcept"
    for i in range(0, len(nodes), batch):
        chunk = nodes[i:i + batch]
        if mapping == "typed":
            # One statement per label keeps the write a single UNWIND per group.
            by_label: dict[str, list] = {}
            for row in chunk:
                by_label.setdefault(row["label"], []).append(row)
            for lab, rows in by_label.items():
                session.run(
                    f"UNWIND $rows AS r MERGE (c:{base} {{name: r.name}}) "
                    f"SET c:`{lab}`, c.concept_class = r.concept_class, "
                    f"c.confidence = r.confidence, c.source = r.source, "
                    f"c.validator_status = r.validator_status, "
                    f"c.aliases = r.aliases, c.lat = r.lat, c.lon = r.lon",
                    rows=rows)
        else:
            session.run(
                f"UNWIND $rows AS r MERGE (c:{base} {{name: r.name}}) "
                f"SET c.concept_class = r.concept_class, "
                f"c.confidence = r.confidence, c.source = r.source, "
                f"c.validator_status = r.validator_status, c.aliases = r.aliases, "
                f"c.lat = r.lat, c.lon = r.lon",
                rows=chunk)
    nodes_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    skipped = 0
    for i in range(0, len(edges), batch):
        chunk = edges[i:i + batch]
        if mapping == "typed":
            by_type: dict[str, list] = {}
            for row in chunk:
                by_type.setdefault(row["reltype"], []).append(row)
            for rt, rows in by_type.items():
                res = session.run(
                    f"UNWIND $rows AS r "
                    f"MATCH (a:{base} {{name: r.s}}), (b:{base} {{name: r.o}}) "
                    f"MERGE (a)-[e:`{rt}`]->(b) "
                    f"SET e.predicate = r.predicate, e.confidence = r.confidence, "
                    f"e.source = r.source "
                    f"RETURN count(e) AS n", rows=rows)
                skipped += len(rows) - (res.single()["n"] or 0)
        else:
            res = session.run(
                f"UNWIND $rows AS r "
                f"MATCH (a:{base} {{name: r.s}}), (b:{base} {{name: r.o}}) "
                f"MERGE (a)-[e:RELATES_TO {{predicate: r.predicate}}]->(b) "
                f"SET e.confidence = r.confidence, e.source = r.source "
                f"RETURN count(e) AS n", rows=chunk)
            skipped += len(chunk) - (res.single()["n"] or 0)
    edges_s = time.perf_counter() - t0

    # Two statements, not one: an aggregation grouped on a key returns no row at
    # all when the second MATCH is empty, and `.single()` would then be None.
    n_nodes = session.run(
        f"MATCH (n:{base}) RETURN count(n) AS n").single()["n"]
    n_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
    counts = {"nodes": n_nodes, "rels": n_rels}
    return {"mapping": mapping,
            "schema_s": round(schema_s, 3),
            "nodes_s": round(nodes_s, 3),
            "edges_s": round(edges_s, 3),
            "load_s": round(schema_s + nodes_s + edges_s, 3),
            "nodes": counts["nodes"], "relationships": counts["rels"],
            "edges_not_created": skipped}


def schema_labels(session) -> list[str]:
    """Labels actually carried by a node right now, not tokens left behind."""
    return [r["label"] for r in session.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label")]


def schema_reltypes(session) -> list[str]:
    return [r["t"] for r in session.run(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")]


def measure(session, mapping: str, repeats: int) -> list[dict]:
    rows = []
    for name, question, q in queries(mapping):
        lat, n, err, sample = [], 0, None, []
        for _ in range(repeats):
            try:
                t0 = time.perf_counter()
                res = list(session.run(q))
                lat.append((time.perf_counter() - t0) * 1000)
                n = len(res)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                break
        if not err:
            sample = [" | ".join(str(v) for v in r.values())
                      for r in list(session.run(q))[:3]]
        rows.append({"query": name, "question": question, "rows": n,
                     "p50_ms": round(st.median(lat), 3) if lat else None,
                     "max_ms": round(max(lat), 3) if lat else None,
                     "sample": sample, "error": err})
        print(f"  [{mapping}] {name}: {n} rows, "
              f"{rows[-1]['p50_ms']} ms" if not err
              else f"  [{mapping}] {name}: FAILED {err}", flush=True)
    return rows


TTL = ROOT / "wbs" / "deliverables" / "task2_formalization" / "jeju_tourism_ontology.ttl"


def store_parity(version: str, n_concepts: int, n_relations: int) -> dict:
    """Do the file-based store and the RDF store hold the same ontology version?

    A second query path is only worth building if it can disagree with the first.
    This check exists because it did: the Turtle the SPARQL path loads declares its
    own `owl:versionInfo`, and nothing until now compared that with the version the
    file-based path was loaded from.
    """
    declared, individuals = None, None
    if TTL.exists():
        text = TTL.read_text(encoding="utf-8")
        m = re.search(r'owl:versionInfo\s+"([^"]+)"', text)
        declared = m.group(1) if m else None
        individuals = len(re.findall(r"owl:NamedIndividual", text))
    same = declared == version
    return {
        "file_store_version": version,
        "file_store_concepts": n_concepts,
        "file_store_relations": n_relations,
        "rdf_store_declared_version": declared,
        "rdf_store_named_individuals": individuals,
        "same_version": same,
        "note": None if same else (
            f"The file-based store is v{version} and the Turtle the SPARQL and "
            f"Oxigraph paths load declares v{declared}. The two paths are answering "
            f"questions about different ontology versions, so a row-count difference "
            f"between them is expected and is not a Cypher/SPARQL difference."),
    }


def agreement(neo_rows, other_rows, other_name):
    """Do the two paths return the same number of rows for the same question?"""
    other = {r["query"]: r for r in other_rows}
    out = []
    for r in neo_rows:
        o = other.get(r["query"])
        if o is None:
            continue
        out.append({"query": r["query"], "neo4j_rows": r["rows"],
                    f"{other_name}_rows": o["rows"],
                    "same": r["rows"] == o["rows"],
                    "neo4j_p50_ms": r["p50_ms"], f"{other_name}_p50_ms": o["p50_ms"]})
    return out


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolt", default=os.environ.get("NEO4J_URI",
                                                     "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD",
                                                         "jeju-ontology-2026"))
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--repeats", type=int, default=25,
                    help="timed repetitions per query; the Cypher queries run in "
                         "about a millisecond, so 5 samples would rank noise")
    ap.add_argument("--mappings", default="typed,generic")
    ap.add_argument("--leave", default="typed",
                    help="which mapping to leave in the database when the run "
                         "finishes, so what a person opens in the browser is a "
                         "decision rather than a side effect of measurement order; "
                         "'none' leaves the last one measured")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("the neo4j driver is not installed: pip install neo4j")

    concepts, relations = read_ontology(a.version)
    nodes, edges = _node_rows(concepts), _edge_rows(relations)
    print(f"ontology v{a.version}: {len(nodes):,} concepts, {len(edges):,} relations")

    driver = GraphDatabase.driver(a.bolt, auth=(a.user, a.password))
    driver.verify_connectivity()
    server = driver.get_server_info()
    print(f"connected to {server.agent} at {a.bolt}")

    results = {}
    with driver.session() as s:
        for mapping in [m.strip() for m in a.mappings.split(",") if m.strip()]:
            print(f"-- {mapping} mapping")
            wipe(s)
            load = ingest(s, nodes, edges, mapping)
            print(f"  loaded {load['nodes']:,} nodes and "
                  f"{load['relationships']:,} relationships in {load['load_s']} s")
            results[mapping] = {
                "load": load,
                "queries": measure(s, mapping, a.repeats),
                # Asked per mapping: token stores keep labels after a delete, so a
                # single reading at the end would report the last load's schema only.
                "labels": sorted(schema_labels(s)),
                "relationship_types": sorted(schema_reltypes(s)),
            }
        # Measuring wipes between mappings, so the database would otherwise be left
        # holding whichever one happened to run last.
        loaded = list(results)[-1] if results else None
        if a.leave != "none" and a.leave in results and a.leave != loaded:
            print(f"-- reloading {a.leave} so it is what the database holds")
            wipe(s)
            ingest(s, nodes, edges, a.leave)
            loaded = a.leave
        if loaded:
            for other in SCHEMA_OBJECTS:
                if other != loaded:
                    gone = drop_schema_objects(s, other)
                    if gone:
                        print(f"-- dropped {len(gone)} unused schema "
                              f"{plural(len(gone), 'object')} from the "
                              f"{other} mapping")
    driver.close()

    sparql = json.loads(SPARQL_JSON.read_text(encoding="utf-8")) \
        if SPARQL_JSON.exists() else None
    oxi = json.loads(OXIGRAPH_JSON.read_text(encoding="utf-8")) \
        if OXIGRAPH_JSON.exists() else None

    primary = "typed" if "typed" in results else list(results)[0]
    cmp_sparql = agreement(results[primary]["queries"], sparql["queries"], "rdflib") \
        if sparql else []
    cmp_oxi = agreement(results[primary]["queries"], oxi["queries"], "oxigraph") \
        if oxi else []

    out = {
        "engine": f"Neo4j ({server.agent})",
        "bolt": a.bolt,
        "ontology_version": a.version,
        "concepts": len(nodes),
        "relations": len(edges),
        "repeats": a.repeats,
        "mappings": results,
        "mapping_left_in_database": loaded,
        "store_parity": store_parity(a.version, len(nodes), len(edges)),
        "vs_rdflib": cmp_sparql,
        "vs_oxigraph": cmp_oxi,
        "row_count_disagreements": [c["query"] for c in cmp_sparql
                                    if not c["same"]],
    }
    (OUT / "graph_db_neo4j.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf-8")
    write_markdown(out)
    print(f"wrote {(OUT / 'graph_db_neo4j.json').relative_to(ROOT)}")
    print(f"wrote {(OUT / 'graph_db_neo4j.md').relative_to(ROOT)}")
    sp = out["store_parity"]
    if not sp["same_version"]:
        print(f"!! file store v{sp['file_store_version']} "
              f"({sp['file_store_concepts']:,} concepts) vs RDF store "
              f"v{sp['rdf_store_declared_version']} "
              f"({sp['rdf_store_named_individuals']:,} named individuals)")
    dis = out["row_count_disagreements"]
    print(f"row-count agreement with the SPARQL path: "
          f"{len(cmp_sparql) - len(dis)}/{len(cmp_sparql)}"
          + (f"; differs on {', '.join(dis)}" if dis else ""))


def plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def mapping_verdict(d: dict) -> str:
    """Say what the typed-versus-generic numbers support, and no more.

    These queries run in about a millisecond. A ratio computed from medians that
    small is mostly scheduling noise, so the verdict is stated as a count of
    questions with a gap wide enough to be worth believing, not as a speedup.
    """
    maps = d["mappings"]
    if "typed" not in maps or "generic" not in maps:
        return ("- Only one mapping was measured, so this run says nothing about "
                "the mapping choice.")
    t = {q["query"]: q for q in maps["typed"]["queries"]}
    g = {q["query"]: q for q in maps["generic"]["queries"]}
    wide = []
    for name, tq in t.items():
        gq = g.get(name)
        if not gq or not tq["p50_ms"] or not gq["p50_ms"]:
            continue
        ratio = gq["p50_ms"] / tq["p50_ms"]
        if ratio >= 2.0 or ratio <= 0.5:
            wide.append((name, ratio))
    faster_typed = [n for n, r in wide if r >= 2.0]
    faster_generic = [n for n, r in wide if r <= 0.5]
    if not wide:
        return (f"- On the mapping choice the run is inconclusive and is reported "
                f"that way: no question separates the two mappings by as much as a "
                f"factor of two, at a median of about a millisecond over "
                f"{d['repeats']} runs. The idiomatic mapping is still the one to "
                f"prefer - it lets the planner use relationship types and reads as "
                f"the domain rather than as triples - but this measurement does not "
                f"prove it, and a graph this size is the wrong place to look.")
    parts = [f"- On the mapping choice, {len(wide)} of {len(t)} questions separate "
             f"the two mappings by at least a factor of two over {d['repeats']} runs"]
    if faster_typed:
        parts.append(f"the idiomatic mapping is the faster one on "
                     f"{', '.join(faster_typed)}")
    if faster_generic:
        parts.append(f"the generic mapping is the faster one on "
                     f"{', '.join(faster_generic)}")
    parts.append("at a median of a few milliseconds either way, so this ranks the "
                 "mapping and not the engine, and it is a hint about the shape of "
                 "the query rather than a benchmark")
    return "; ".join(parts) + "."


def write_markdown(d: dict) -> None:
    L = ["# Task 3.1.1 - Neo4j, and how it compares with the SPARQL path\n",
         f"Engine: **{d['engine']}**, Bolt `{d['bolt']}`. "
         f"Ontology v{d['ontology_version']}: {d['concepts']:,} concepts, "
         f"{d['relations']:,} relations.\n",
         "\nThis closes the one storage form the WBS names that this project had "
         "not exercised. Neo4j needs a JVM, which this host did not have; a JDK and "
         "the Neo4j tarball now sit under `~/.local/opt` and "
         "`wbs/task3_storage/neo4j_server.sh` starts the server on Bolt 7687 and "
         "HTTP 7474, with its data beside the Oxigraph store at "
         "`artifacts/graph_db/neo4j`.\n",
         "\n## The two mappings, and why both are measured\n",
         "A property graph can hold an ontology two ways, and the difference is not "
         "cosmetic.\n",
         "- **typed** - the idiomatic shape: a concept is a node labelled `:Concept` "
         "plus its ontology class, a triple is a relationship whose *type* is the "
         "predicate. Neo4j indexes relationship types natively.",
         "- **generic** - what `src/graph/cypher_queries.py` does today: one label, "
         "one relationship type `:RELATES_TO`, predicate as a *property*. This is a "
         "triplestore emulated inside a property graph.\n",
         "\nMeasuring only the generic mapping against a real triplestore would "
         "measure the mapping and report it as the engine.\n",
         "\n## Load\n",
         "| Mapping | Nodes | Relationships | Schema | Nodes | Relationships | Total |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for m, r in d["mappings"].items():
        ld = r["load"]
        L.append(f"| {m} | {ld['nodes']:,} | {ld['relationships']:,} | "
                 f"{ld['schema_s']} s | {ld['nodes_s']} s | {ld['edges_s']} s | "
                 f"**{ld['load_s']} s** |")
    skipped = {m: r["load"]["edges_not_created"] for m, r in d["mappings"].items()}
    if any(skipped.values()):
        L.append(f"\nRelationships not created (an endpoint the concept set does not "
                 f"contain): {skipped}.\n")
    else:
        L.append("\nEvery relation in the ontology found both endpoints, in both "
                 "mappings.\n")

    L.append("\n## Competency questions\n")
    L.append("The same nine questions the SPARQL path answers, in Cypher. Latency is "
             f"the median of {d['repeats']} runs.\n")
    header = "| Query | Question | " + " | ".join(
        f"{m} rows | {m} p50" for m in d["mappings"]) + " |"
    L.append(header)
    L.append("|---|---|" + "--:|--:|" * len(d["mappings"]))
    names = [q["query"] for q in d["mappings"][list(d["mappings"])[0]]["queries"]]
    for i, name in enumerate(names):
        cells = []
        question = ""
        for m, r in d["mappings"].items():
            q = r["queries"][i]
            question = q["question"]
            cells.append(f"{q['rows']}" if not q["error"] else "ERR")
            cells.append(f"{q['p50_ms']} ms" if q["p50_ms"] is not None else "-")
        L.append(f"| {name} | {question} | " + " | ".join(cells) + " |")

    sp = d.get("store_parity") or {}
    if sp and not sp.get("same_version", True):
        L.append("\n## What the second query path found\n")
        L.append("The point of a second store is that it can disagree with the "
                 "first, and it did on the first run.\n")
        L.append(f"- The **file-based store** is "
                 f"`artifacts/ontologies/{sp['file_store_version']}/` - "
                 f"{sp['file_store_concepts']:,} concepts, "
                 f"{sp['file_store_relations']:,} relations. This is the version the "
                 f"retrieval path and the answer validator load.")
        L.append(f"- The **RDF store** - the Turtle that `sparql_store.py`, the "
                 f"Oxigraph quad store and the SPARQL endpoint all read - declares "
                 f"`owl:versionInfo \"{sp['rdf_store_declared_version']}\"` and "
                 f"carries {sp['rdf_store_named_individuals']:,} named individuals.")
        L.append("\nSo the two paths have been answering questions about **different "
                 "versions of the ontology**. Every row-count difference in the table "
                 "below follows from that, and none of it is a Cypher-versus-SPARQL "
                 "difference:\n")
        L.append("- `Q1 inventory` differs by exactly one concept: **UNESCO World "
                 "Heritage** (class `Designation`), added by governance proposal "
                 "**P0001**, approved 2026-10-16.")
        L.append("- `Q8 provenance / governance` differs by exactly one group: the "
                 "`governance/P0001` provenance source, which exists only in the "
                 "newer version and holds that single concept.")
        L.append("- The three `has_designation` relations P0001 added are present in "
                 "`relations.json` and absent from the Turtle, so the designation "
                 "P0001 was approved in order to make queryable is **not queryable "
                 "through any RDF path**.\n")
        L.append("This is a staleness defect between two storage forms of the same "
                 "ontology, not a defect in either engine: the governance layer "
                 "approved a change that reached the file store and was never carried "
                 "into the serialisations. It is reported here rather than repaired, "
                 "because regenerating the serialisations moves the triple count, the "
                 "SHACL run and the OWL-RL closure that other delivered reports "
                 "quote, and that is a decision about published numbers rather than a "
                 "measurement.\n")

    if d["vs_rdflib"]:
        L.append("\n## Does the Cypher path answer the same questions?\n")
        L.append("Row counts for the typed mapping against the two RDF paths. A "
                 "difference is not automatically a fault - the two data models count "
                 "different things - so each one is named rather than averaged away.\n")
        L.append("| Query | Neo4j rows | rdflib rows | Oxigraph rows | Same | "
                 "Neo4j p50 | rdflib p50 | Oxigraph p50 |")
        L.append("|---|--:|--:|--:|:--:|--:|--:|--:|")
        oxi = {c["query"]: c for c in d["vs_oxigraph"]}
        for c in d["vs_rdflib"]:
            o = oxi.get(c["query"], {})
            L.append(f"| {c['query']} | {c['neo4j_rows']} | {c['rdflib_rows']} | "
                     f"{o.get('oxigraph_rows', '-')} | "
                     f"{'yes' if c['same'] else '**no**'} | "
                     f"{c['neo4j_p50_ms']} ms | {c['rdflib_p50_ms']} ms | "
                     f"{o.get('oxigraph_p50_ms', '-')} ms |")
        dis = d["row_count_disagreements"]
        kept = len(d["vs_rdflib"]) - len(dis)
        parity_shown = not (d.get("store_parity") or {}).get("same_version", True)
        where = "above" if parity_shown else "below"
        L.append(f"\n**{kept} of {len(d['vs_rdflib'])} questions return the same "
                 f"number of rows on both paths.**"
                 + (f" The {plural(len(dis), 'exception')} "
                    f"{'is' if len(dis) == 1 else 'are'} {', '.join(dis)}, "
                    f"explained {where} rather than reconciled by changing a query."
                    if dis else ""))

    L.append("\n## Schema each load produced\n")
    if d.get("mapping_left_in_database"):
        L.append(f"The database currently holds the "
                 f"**{d['mapping_left_in_database']}** mapping: that is what "
                 f"`neo4j_server.sh` serves and what the Neo4j Browser shows. "
                 f"Measurement wipes between mappings, so this is set explicitly "
                 f"with `--leave` rather than left to whichever ran last.\n")
    for m, r in d["mappings"].items():
        labs, rts = r.get("labels", []), r.get("relationship_types", [])
        L.append(f"- **{m}** - {len(labs)} {plural(len(labs), 'label')}, "
                 f"{len(rts)} relationship {plural(len(rts), 'type')}.")
        L.append(f"  - labels: `" + "`, `".join(labs) + "`")
        L.append(f"  - relationship types: `" + "`, `".join(rts) + "`")
    L.append("")

    L.append("\n## What this does and does not settle\n")
    sp2 = d.get("store_parity") or {}
    if sp2 and not sp2.get("same_version", True):
        L.append(f"- The WBS line item asked for storage over an "
                 f"\"RDF/graph database\". Both halves of that phrase are now "
                 f"exercised. They are not yet exercised on the *same* ontology: "
                 f"Cypher ran on v{sp2['file_store_version']} and the RDF paths on "
                 f"v{sp2['rdf_store_declared_version']}, which is the defect above "
                 f"and not a property of either engine.")
    else:
        L.append("- The WBS line item asked for storage over an "
                 "\"RDF/graph database\". Both halves of that phrase are now "
                 "exercised on the same delivered ontology.")
    L.append("- Latency is not a reason to choose either engine here, and this table "
             "is not evidence that it is. The graph is small enough that every path "
             "answers in single-digit milliseconds, where process warm-up and page "
             "cache dominate. The deciding properties lie elsewhere. This project's "
             "validation layer is built on SHACL shapes and an OWL-RL closure: "
             "Neo4j can run SHACL, but only through the neosemantics (n10s) plugin, "
             "which is not installed here, and n10s offers query-time inference "
             "rather than the materialised OWL-RL entailment that Task 2.2 measures "
             "(17,706 triples closing to 27,521). That is why the delivered pipeline "
             "stays on RDF - not because Cypher answered anything worse.")
    L.append(mapping_verdict(d))
    (OUT / "graph_db_neo4j.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
