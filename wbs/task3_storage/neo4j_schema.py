"""
WBS Task 3.1.1 - read the full ontology schema back out of Neo4j.

`neo4j_store.py` loads the ontology and times queries against it. This module
answers a different question: what does the schema actually look like once it is
in there? It reads the live database rather than the source JSON, so what it
prints is what the server holds - which is the only way to check that a load did
what it was supposed to do.

It reports, all from the live store: every label with its node count, every
relationship type with its count and the label pairs it connects, property keys
per label, constraints and indexes, the class hierarchy under `subclass_of`, and
the governance-added elements that the RDF serialisations are missing.

  bash wbs/task3_storage/neo4j_server.sh start
  python -m wbs.task3_storage.neo4j_schema
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "wbs" / "deliverables" / "task3_storage"


def rows(session, cypher: str, **params) -> list[dict]:
    return [dict(r) for r in session.run(cypher, **params)]


def collect(session) -> dict:
    labels = rows(session, """
        MATCH (n) UNWIND labels(n) AS label
        RETURN label, count(*) AS nodes ORDER BY nodes DESC, label""")

    reltypes = rows(session, """
        MATCH (a)-[r]->(b)
        RETURN type(r) AS type, count(*) AS edges,
               count(DISTINCT a) AS distinct_subjects,
               count(DISTINCT b) AS distinct_objects
        ORDER BY edges DESC, type""")

    # Which classes a predicate actually connects - the domain/range the data has,
    # as opposed to the one the schema document declares.
    endpoints = rows(session, """
        MATCH (a:Concept)-[r]->(b:Concept)
        RETURN type(r) AS type, a.concept_class AS subject_class,
               b.concept_class AS object_class, count(*) AS edges
        ORDER BY type, edges DESC""")

    props = rows(session, """
        MATCH (n:Concept) UNWIND keys(n) AS key
        RETURN key, count(*) AS nodes,
               count(DISTINCT n.concept_class) AS classes
        ORDER BY nodes DESC, key""")

    rel_props = rows(session, """
        MATCH ()-[r]->() UNWIND keys(r) AS key
        RETURN key, count(*) AS relationships ORDER BY relationships DESC, key""")

    constraints = rows(session, "SHOW CONSTRAINTS YIELD name, type, "
                                "labelsOrTypes, properties "
                                "RETURN name, type, labelsOrTypes, properties")
    indexes = rows(session, "SHOW INDEXES YIELD name, type, labelsOrTypes, "
                            "properties, state "
                            "RETURN name, type, labelsOrTypes, properties, state")

    hierarchy = rows(session, """
        MATCH (c:Concept)-[:SUBCLASS_OF]->(p:Concept)
        RETURN p.name AS parent, collect(c.name)[0..12] AS children,
               count(c) AS n ORDER BY n DESC, parent LIMIT 25""")

    depth = rows(session, """
        MATCH path = (c:Concept)-[:SUBCLASS_OF*1..]->(root:Concept)
        WHERE NOT (root)-[:SUBCLASS_OF]->()
        RETURN max(length(path)) AS max_depth, avg(length(path)) AS mean_depth,
               count(DISTINCT c) AS classified""")

    # The elements governance added, which no RDF path can see (see graph_db_neo4j.md).
    governance = rows(session, """
        MATCH (c:Concept) WHERE c.source STARTS WITH 'governance/'
        OPTIONAL MATCH (c)<-[r]-(other)
        RETURN c.name AS name, c.concept_class AS class, c.source AS source,
               count(r) AS inbound_relations,
               collect(DISTINCT other.name)[0..5] AS from_entities""")

    orphans = rows(session, """
        MATCH (n:Concept) WHERE NOT (n)--() RETURN count(n) AS isolated""")

    degree = rows(session, """
        MATCH (n:Concept) WITH n, count{(n)--()} AS deg
        RETURN n.name AS name, n.concept_class AS class, deg
        ORDER BY deg DESC LIMIT 10""")

    return {"labels": labels, "relationship_types": reltypes,
            "endpoint_classes": endpoints, "node_properties": props,
            "relationship_properties": rel_props, "constraints": constraints,
            "indexes": indexes, "hierarchy": hierarchy,
            "hierarchy_depth": depth[0] if depth else {},
            "governance_elements": governance,
            "isolated": orphans[0]["isolated"] if orphans else None,
            "top_degree": degree}


def markdown(d: dict, meta: dict) -> str:
    L = [f"# Task 3.1.1 - the ontology schema as Neo4j holds it\n",
         f"Read back from the live server, not from the source files: "
         f"**{meta['engine']}** at `{meta['bolt']}`, database `{meta['database']}`, "
         f"ontology v{meta['version']}.\n",
         f"\n{meta['nodes']:,} nodes and {meta['relationships']:,} relationships.\n"]

    L += ["\n## Labels\n",
          f"{len(d['labels'])} labels. `Concept` is carried by every node; the rest "
          f"are the ontology classes, one label each.\n",
          "| Label | Nodes |", "|---|--:|"]
    for r in d["labels"]:
        L.append(f"| `{r['label']}` | {r['nodes']:,} |")

    L += ["\n## Relationship types\n",
          f"{len(d['relationship_types'])} types, one per ontology predicate.\n",
          "| Type | Edges | Distinct subjects | Distinct objects |",
          "|---|--:|--:|--:|"]
    for r in d["relationship_types"]:
        L.append(f"| `{r['type']}` | {r['edges']:,} | "
                 f"{r['distinct_subjects']:,} | {r['distinct_objects']:,} |")

    L += ["\n## What each predicate actually connects\n",
          "The domain and range the *data* has, which is not the same claim as the "
          "one the schema document makes. Only pairs with at least 10 edges are "
          "listed.\n",
          "| Type | Subject class | Object class | Edges |", "|---|---|---|--:|"]
    for r in d["endpoint_classes"]:
        if r["edges"] >= 10:
            L.append(f"| `{r['type']}` | {r['subject_class']} | "
                     f"{r['object_class']} | {r['edges']:,} |")

    L += ["\n## Properties\n", "| Node property | On nodes | Across classes |",
          "|---|--:|--:|"]
    for r in d["node_properties"]:
        L.append(f"| `{r['key']}` | {r['nodes']:,} | {r['classes']} |")
    L += ["", "| Relationship property | On relationships |", "|---|--:|"]
    for r in d["relationship_properties"]:
        L.append(f"| `{r['key']}` | {r['relationships']:,} |")

    L += ["\n## Constraints and indexes\n", "| Name | Kind | On | Properties |",
          "|---|---|---|---|"]
    for r in d["constraints"]:
        L.append(f"| `{r['name']}` | constraint / {r['type']} | "
                 f"{r['labelsOrTypes']} | {r['properties']} |")
    for r in d["indexes"]:
        L.append(f"| `{r['name']}` | index / {r['type']} ({r['state']}) | "
                 f"{r['labelsOrTypes']} | {r['properties']} |")

    dep = d["hierarchy_depth"]
    if dep:
        L += ["\n## Class hierarchy\n",
              f"Maximum depth {dep.get('max_depth')}, mean "
              f"{dep.get('mean_depth'):.3f} over {dep.get('classified'):,} "
              f"classified concepts, measured along `SUBCLASS_OF` to a root.\n"
              if dep.get("mean_depth") is not None else "",
              "| Parent | Children | Shown |", "|---|--:|---|"]
        for r in d["hierarchy"]:
            L.append(f"| `{r['parent']}` | {r['n']:,} | "
                     + ", ".join(f"`{c}`" for c in r["children"]) + " |")

    L.append(f"\n## Connectivity\n")
    L.append(f"Isolated concepts: **{d['isolated']}**.\n")
    L += ["| Most connected concept | Class | Degree |", "|---|---|--:|"]
    for r in d["top_degree"]:
        L.append(f"| `{r['name']}` | {r['class']} | {r['deg']:,} |")

    if d["governance_elements"]:
        L += ["\n## The governance-added elements\n",
              "These exist in the file-based store and in Neo4j, and in **none** of "
              "the RDF paths: the serialisations declare an older version. "
              "`graph_db_neo4j.md` carries the detail. This table is the check that "
              "they really are reachable here.\n",
              "| Concept | Class | Source | Inbound | From |",
              "|---|---|---|--:|---|"]
        for r in d["governance_elements"]:
            L.append(f"| `{r['name']}` | {r['class']} | `{r['source']}` | "
                     f"{r['inbound_relations']} | "
                     + ", ".join(f"`{n}`" for n in r["from_entities"] if n) + " |")

    L.append("\n## Opening this in the Neo4j Browser\n")
    L.append(f"The server listens on localhost only, so from another machine open an "
             f"SSH tunnel first:\n")
    L.append("```\nssh -L 7474:127.0.0.1:7474 -L 7687:127.0.0.1:7687 "
             f"{meta['user_hint']}\n```\n")
    L.append(f"Then open `http://localhost:7474`, connect to "
             f"`bolt://127.0.0.1:7687` and sign in.\n")
    # Worth stating plainly: the Browser's defaults truncate this graph without
    # announcing it, so "I ran the query and saw 300 nodes" is the expected
    # first experience unless the caps are raised.
    L.append("**Raise the render caps before drawing the whole graph.** The "
             "Neo4j Browser renders the first 300 nodes of a result and at most "
             "100 neighbours per node, and says nothing about what it dropped; "
             f"with {meta['nodes']:,} nodes and {meta['relationships']:,} "
             "relationships here, the defaults show under a fifth of the "
             "ontology.\n")
    L.append("```\n:config initialNodeDisplay: 2000, maxNeighbours: 9000\n```\n")
    L.append("Then the whole ontology in one canvas:\n")
    L.append("```cypher\nMATCH (n:Concept)-[r]->(m:Concept)\nRETURN n, r, m;\n```\n")
    L.append("`whole_ontology.cypher`, beside this file, is the full ordered "
             "script - the same ten queries the explorer's **Neo4j** tab shows, "
             "generated from `neo4j_export.BROWSER_SCRIPT` so the two cannot "
             "disagree.\n")
    L.append("Other useful starting queries:\n")
    L.append("""```cypher
// the schema itself, drawn
CALL db.schema.visualization();

// every class and how many instances it has
MATCH (n:Concept) RETURN n.concept_class AS class, count(*) AS n ORDER BY n DESC;

// one class, with everything it connects to
MATCH (b:Beach)-[r]-(x) RETURN b, r, x LIMIT 100;

// the governance-added concept the RDF paths cannot see
MATCH (d:Designation {name: 'UNESCO World Heritage'})<-[r]-(x) RETURN d, r, x;

// region containment, any depth
MATCH p = (c:Concept)-[:PART_OF*1..]->(:Concept {name: 'Jeju'}) RETURN p LIMIT 50;

// safety constraints, the rules the validator enforces
MATCH (s)-[r:HAS_RISK|AVOID_DURING|REQUIRES]->(o) RETURN s, r, o LIMIT 100;
```""")
    return "\n".join(x for x in L if x != "") + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolt", default=os.environ.get("NEO4J_URI",
                                                     "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD",
                                                         "jeju-ontology-2026"))
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--user-hint", default="tahirahmad@203.253.207.64")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("the neo4j driver is not installed: pip install neo4j")

    driver = GraphDatabase.driver(a.bolt, auth=(a.user, a.password))
    driver.verify_connectivity()
    info = driver.get_server_info()
    with driver.session(database=a.database) as s:
        n_nodes = rows(s, "MATCH (n) RETURN count(n) AS n")[0]["n"]
        n_rels = rows(s, "MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
        if n_nodes == 0:
            driver.close()
            sys.exit("the database is empty; run "
                     "python -m wbs.task3_storage.neo4j_store first")
        d = collect(s)
    driver.close()

    meta = {"engine": f"Neo4j {info.agent.split('/')[-1]}", "bolt": a.bolt,
            "database": a.database, "version": a.version, "nodes": n_nodes,
            "relationships": n_rels, "user_hint": a.user_hint}
    (OUT / "ontology_schema_neo4j.json").write_text(
        json.dumps({"meta": meta, **d}, indent=2, default=str), encoding="utf-8")
    (OUT / "ontology_schema_neo4j.md").write_text(markdown(d, meta),
                                                  encoding="utf-8")
    print(f"{n_nodes:,} nodes, {n_rels:,} relationships, "
          f"{len(d['labels'])} labels, {len(d['relationship_types'])} "
          f"relationship types")
    if d["governance_elements"]:
        print(f"governance-added elements present here and absent from RDF: "
              f"{', '.join(r['name'] for r in d['governance_elements'])}")
    print(f"wrote {(OUT / 'ontology_schema_neo4j.md').relative_to(ROOT)}")
    print(f"wrote {(OUT / 'ontology_schema_neo4j.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
