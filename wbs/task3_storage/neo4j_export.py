"""WBS Task 3.1.1 - read the complete ontology back out of Neo4j, over Cypher.

The WBS names Neo4j as the ontology store, so the figures and the explorer must
be able to source their data from the running server rather than from the JSON
files the server was loaded from. This module is that source: it runs Cypher
against a live Neo4j, returns concepts and relations in exactly the shape
`neo4j_store.read_ontology` returns, and verifies the result against the file
store so "sourced from Neo4j" is a checked claim and not a label.

Fidelity matters here. The labels and relationship types are sanitised on the way
in (`Travel Constraint` -> `Travel_Constraint`, `has-risk` -> `HAS_RISK`), so
reconstructing a predicate from a relationship type would be lossy. The ingest
also stores the unsanitised `concept_class` and `predicate` as properties, and
this exporter reads those - which is what makes the round trip exact.

    python -m wbs.task3_storage.neo4j_export            # export + verify
    python -m wbs.task3_storage.neo4j_export --no-verify
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIV = ROOT / "wbs" / "deliverables" / "task3_storage"
SNAPSHOT = DELIV / "ontology_from_neo4j.json"

# Every view built from this module can print the query that produced it, so a
# reader can re-run it in the Neo4j Browser and get the same rows back.
CONCEPT_CYPHER = """\
MATCH (n:Concept)
RETURN n.name              AS name,
       n.concept_class     AS concept_class,
       n.aliases           AS aliases,
       n.confidence        AS confidence,
       n.source            AS source,
       n.validator_status  AS validator_status,
       n.lat               AS lat,
       n.lon               AS lon,
       labels(n)           AS labels
ORDER BY n.name"""

RELATION_CYPHER = """\
MATCH (s:Concept)-[r]->(o:Concept)
RETURN s.name       AS subject,
       r.predicate  AS predicate,
       o.name       AS object,
       type(r)      AS reltype,
       r.confidence AS confidence,
       r.source     AS source
ORDER BY s.name, r.predicate, o.name"""

COUNT_CYPHER = """\
MATCH (n) WITH count(n) AS nodes
MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"""


# The queries a reader runs in the Neo4j Browser to see the whole ontology there.
# They live here rather than in the HTML so the page, the .cypher script and the
# schema report cannot drift apart.
#
# The first entry is not optional. The Neo4j Browser renders the first 300 nodes
# of a result and 100 neighbours per node by default, so pasting "return
# everything" draws under a fifth of this graph and says nothing about the rest;
# a reader would reasonably conclude the ontology is that small.
# Which schema the database is actually holding. The project can serve the
# ontology under either of two mappings and Neo4j Community has one database, so
# a query naming `[:SUBCLASS_OF]` returns nothing at all when the generic mapping
# is resident - silently, as an empty result rather than an error. Every query
# below is therefore chosen after asking the server which mapping it has.
def detect_mapping(session) -> dict:
    labels = [r["label"] for r in session.run("CALL db.labels() YIELD label "
                                              "RETURN label ORDER BY label")]
    types = [r["t"] for r in session.run("CALL db.relationshipTypes() "
                                         "YIELD relationshipType AS t "
                                         "RETURN t ORDER BY t")]
    # Token stores keep labels and types after their nodes are deleted, so the
    # declared list is not evidence; count what is in use.
    live_labels = [r["l"] for r in session.run(
        "MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l ORDER BY l")]
    live_types = [r["t"] for r in session.run(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t ORDER BY t")]
    generic = (live_types == ["RELATES_TO"])
    return {
        "name": "generic" if generic else "typed",
        "node_label": "Concept",
        "relationship_type": "RELATES_TO" if generic else "<one per predicate>",
        "labels_in_use": live_labels,
        "relationship_types_in_use": live_types,
        "labels_declared": labels,
        "relationship_types_declared": types,
        "description": (
            "one :Concept label and one [:RELATES_TO] type carrying the "
            "predicate as a property - the schema the companion project at "
            "Ontology_Project uses, loaded by neo4j_ontology.py"
            if generic else
            "a label per ontology class and a relationship type per predicate, "
            "loaded by neo4j_store.py"),
    }


def browser_script(mapping: dict) -> list[tuple[str, str, str]]:
    """The Browser queries, written for whichever mapping is resident."""
    generic = mapping["name"] == "generic"
    rel = "[r:RELATES_TO]" if generic else "[r]"
    hier = ("[r:RELATES_TO*1..3]" if generic else "[:SUBCLASS_OF*1..3]")
    hier_where = ("\nWHERE all(x IN relationships(p) WHERE "
                  "x.predicate = 'subclass_of')" if generic else "")
    designation = (
        "MATCH (e:Concept)-[r:RELATES_TO {predicate: 'has_designation'}]->"
        "(d:Concept)\nRETURN d, r, e" if generic else
        "MATCH (d:Designation)<-[r:HAS_DESIGNATION]-(e:Concept)\n"
        "RETURN d, r, e")
    reltype_row = ("RETURN r.predicate AS predicate, count(*) AS edges"
                   if generic else
                   "RETURN type(r) AS reltype, r.predicate AS predicate, "
                   "count(*) AS edges")
    out = list(BROWSER_SCRIPT)
    swap = {
        "The whole ontology":
            f"MATCH (n:Concept)-{rel}->(m:Concept)\nRETURN n, r, m",
        "Count what is there":
            "MATCH (n:Concept) WITH count(n) AS concepts\n"
            f"MATCH (:Concept)-{rel}->(:Concept)\n"
            "RETURN concepts, count(r) AS relations",
        "Predicates by size":
            f"MATCH (:Concept)-{rel}->(:Concept)\n{reltype_row}\n"
            "ORDER BY edges DESC",
        "The Jeju hub":
            "MATCH (n:Concept {name: 'Jeju'})-"
            + ("[r:RELATES_TO]" if generic else "[r]")
            + "-(m:Concept)\nRETURN n, r, m",
        "The class hierarchy":
            f"MATCH p = (c:Concept)-{hier}->(root:Concept){hier_where}\n"
            "RETURN p",
        "Governance-approved elements": designation,
    }
    return [(title, why, swap.get(title, cypher))
            for title, why, cypher in out]


BROWSER_SCRIPT = [
    (":config",
     "Raise the render caps first - the Browser draws 300 nodes and 100 "
     "neighbours per node by default, which silently hides most of this "
     "ontology.",
     ":config initialNodeDisplay: 2000, maxNeighbours: 9000"),
    ("The whole ontology",
     "Every concept and every relation in one canvas. This is the heaviest "
     "query here; give it a few seconds.",
     "MATCH (n:Concept)-[r]->(m:Concept)\nRETURN n, r, m"),
    ("The schema",
     "Neo4j's own schema picture: one node per label, one arrow per "
     "relationship type present in the data.",
     "CALL db.schema.visualization()"),
    ("Count what is there",
     "The two numbers every view of this ontology is built on.",
     "MATCH (n:Concept) WITH count(n) AS concepts\n"
     "MATCH (:Concept)-[r]->(:Concept)\nRETURN concepts, count(r) AS relations"),
    ("Classes by size",
     "Every ontology class and how many instances it holds.",
     "MATCH (n:Concept)\nRETURN n.concept_class AS class, count(*) AS n\n"
     "ORDER BY n DESC"),
    ("Predicates by size",
     "Every relationship type, its ontology predicate, and how many edges use "
     "it.",
     "MATCH (:Concept)-[r]->(:Concept)\n"
     "RETURN type(r) AS reltype, r.predicate AS predicate, count(*) AS edges\n"
     "ORDER BY edges DESC"),
    ("The Jeju hub",
     "The most connected concept in the graph and everything one hop from it.",
     "MATCH (n:Concept {name: 'Jeju'})-[r]-(m:Concept)\nRETURN n, r, m"),
    ("The class hierarchy",
     "Every subclass_of path up to three hops - 50 of them over the 31 "
     "classified classes, because a class three levels down contributes a path "
     "to each ancestor.",
     "MATCH p = (c:Concept)-[:SUBCLASS_OF*1..3]->(root:Concept)\nRETURN p"),
    ("Governance-approved elements",
     "What the RDF serialisations are missing: the UNESCO designation approved "
     "in proposal P0001.",
     "MATCH (d:Designation)<-[r:HAS_DESIGNATION]-(e:Concept)\nRETURN d, r, e"),
    ("Low-confidence concepts",
     "Where the extraction was least sure - the review queue, worst first.",
     "MATCH (n:Concept)\nWHERE n.confidence < 0.7\n"
     "RETURN n.name, n.concept_class, n.confidence, n.source\n"
     "ORDER BY n.confidence ASC LIMIT 50"),
]


def browser_script_text(meta: dict, mapping: dict = None,
                        blocks: list[dict] = None) -> str:
    """The same queries as a file a reader can open and paste from, in order."""
    blocks = blocks or [{"title": a, "why": b, "cypher": c}
                        for a, b, c in BROWSER_SCRIPT]
    bolt = meta["bolt"]
    http = bolt.replace("bolt://", "").rsplit(":", 1)[0] + ":7474"
    L = [f"// Jeju Tourism Ontology - the whole graph in the Neo4j Browser",
         f"// {meta['engine']} {meta.get('edition', '')} at {bolt}, "
         f"database {meta['database']}",
         f"// {meta['nodes']:,} nodes, {meta['relationships']:,} relationships, "
         f"read {meta['exported_utc']}",
         "//",
         "// The server is bound to localhost, so tunnel first:",
         f"//   ssh -L 7474:{http} -L 7687:{bolt.replace('bolt://', '')} "
         f"{meta.get('user_hint', 'you@host')}",
         f"// then open http://{http} and connect to {bolt} as user neo4j.",
         "//",
         "// Paste the blocks below in order. Run the :config line first.",
         ""]
    if mapping:
        L[4:4] = [f"// schema in the database: {mapping['name']} - "
                  f"{mapping['description']}",
                  f"//   {len(mapping['labels_in_use'])} label(s) in use, "
                  f"{len(mapping['relationship_types_in_use'])} "
                  f"relationship type(s)",
                  "//   these queries are written for that mapping; the other "
                  "one needs different ones",
                  "//"]
    for i, (title, why, cypher) in enumerate(
            [(b["title"], b["why"], b["cypher"]) for b in blocks], 1):
        probe = ""
        for b in blocks:
            if b["title"] == title and b.get("rows") is not None:
                probe = f"   [{b['rows']:,} rows, {b['ms']:.0f} ms when generated]"
            elif b["title"] == title and b.get("note") and \
                    b["note"] != "Browser client command, not Cypher":
                probe = f"   [{b['note']}]"
        L.append(f"// {i}. {title}{probe}")
        for line in why.split(". "):
            if line.strip():
                L.append(f"//    {line.strip().rstrip('.')}.")
        L.append(cypher if cypher.startswith(":") else cypher + ";")
        L.append("")
    return "\n".join(L)


def server_meta(session, uri: str, database: str,
                user_hint: str = "tahirahmad@203.253.207.64") -> dict:
    rec = session.run(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions[0] AS version, edition").single()
    counts = session.run(COUNT_CYPHER).single()
    return {
        "engine": f"{rec['name']} {rec['version']}",
        "edition": rec["edition"],
        "bolt": uri,
        "database": database,
        "nodes": counts["nodes"],
        "relationships": counts["relationships"],
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # For the tunnel line a reader has to run; the server is bound to
        # localhost, so the address alone is not enough to reach it.
        "user_hint": user_hint,
    }


def read_concepts(session) -> list[dict]:
    """Concepts in the file store's own shape, assembled from Cypher rows."""
    out = []
    for r in session.run(CONCEPT_CYPHER):
        attrs = {}
        if r["lat"] is not None:
            attrs["lat"] = r["lat"]
        if r["lon"] is not None:
            attrs["lon"] = r["lon"]
        out.append({
            "name": r["name"],
            "concept_class": r["concept_class"],
            "aliases": list(r["aliases"] or []),
            "attributes": attrs,
            "provenance": {
                "confidence": r["confidence"],
                "extraction_prompt_version": r["source"],
                "validator_status": r["validator_status"],
            },
            # Kept so a view can show what Neo4j itself calls this node.
            "neo4j_labels": sorted(r["labels"]),
        })
    return out


def read_relations(session) -> list[dict]:
    out = []
    for r in session.run(RELATION_CYPHER):
        out.append({
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "provenance": {
                "confidence": r["confidence"],
                "extraction_prompt_version": r["source"],
            },
            "neo4j_reltype": r["reltype"],
        })
    return out


def verify(concepts: list[dict], relations: list[dict], version: str) -> dict:
    """Is what came out of Neo4j the same ontology that went in?

    Compared as sets of names and of (subject, predicate, object) triples, so the
    answer does not depend on row order. Anything missing or extra is named, not
    just counted, because a count that happens to match hides a swap.
    """
    from wbs.task3_storage.neo4j_store import read_ontology
    src_c, src_r = read_ontology(version)

    want_n = {c["name"] for c in src_c}
    got_n = {c["name"] for c in concepts}
    want_e = {(r["subject"], r["predicate"], r["object"]) for r in src_r}
    got_e = {(r["subject"], r["predicate"], r["object"]) for r in relations}

    # Classes have to survive too: a node that came back as the generic
    # `Concept` instead of `Beach` would still match on name alone.
    src_cls = {c["name"]: (c.get("concept_class") or "Concept") for c in src_c}
    got_cls = {c["name"]: (c.get("concept_class") or "Concept") for c in concepts}
    wrong_class = sorted(n for n in want_n & got_n if src_cls[n] != got_cls[n])

    return {
        "file_store_version": version,
        "file_store_concepts": len(src_c),
        "file_store_relations": len(src_r),
        "neo4j_concepts": len(concepts),
        "neo4j_relations": len(relations),
        "missing_concepts": sorted(want_n - got_n)[:20],
        "extra_concepts": sorted(got_n - want_n)[:20],
        "missing_relations": [list(t) for t in sorted(want_e - got_e)[:20]],
        "extra_relations": [list(t) for t in sorted(got_e - want_e)[:20]],
        "concepts_with_a_different_class": wrong_class[:20],
        "exact": (want_n == got_n and want_e == got_e and not wrong_class),
    }


def export(uri: str, user: str, password: str, database: str = "neo4j",
           version: str = "1.4.0", do_verify: bool = True,
           user_hint: str = "tahirahmad@203.253.207.64") -> dict:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as s:
            meta = server_meta(s, uri, database, user_hint)
            mapping = detect_mapping(s)
            meta["mapping"] = mapping["name"]
            concepts = read_concepts(s)
            relations = read_relations(s)
            blocks = [{"title": a, "why": b, "cypher": c}
                      for a, b, c in browser_script(mapping)]
            blocks = probe_browser_script(s, blocks)
    finally:
        driver.close()

    # The server's own count(n) must agree with the rows we actually collected;
    # if a node carries no :Concept label the two diverge, and silently.
    unreached_nodes = meta["nodes"] - len(concepts)
    unreached_rels = meta["relationships"] - len(relations)

    out = {
        "source": "neo4j",
        "version": version,
        "meta": meta,
        "cypher": {"concepts": CONCEPT_CYPHER, "relations": RELATION_CYPHER,
                   "counts": COUNT_CYPHER},
        "mapping": mapping,
        "browser_script": blocks,
        "nodes_not_returned_by_the_concept_query": unreached_nodes,
        "relationships_not_returned_by_the_relation_query": unreached_rels,
        "concepts": concepts,
        "relations": relations,
    }
    if do_verify:
        out["verification"] = verify(concepts, relations, version)
    return out


def probe_browser_script(session, blocks: list[dict]) -> list[dict]:
    """Run every shipped query and record what it returns.

    A Cypher query written for the wrong mapping does not fail - it returns an
    empty result, which reads as "the ontology does not contain this". The only
    way to ship a script that is known to work is to run it.
    """
    out = []
    for b in blocks:
        if b["cypher"].startswith(":"):
            out.append(dict(b, rows=None, ms=None,
                            note="Browser client command, not Cypher"))
            continue
        t0 = time.perf_counter()
        try:
            n = sum(1 for _ in session.run(b["cypher"]))
            out.append(dict(b, rows=n,
                            ms=round((time.perf_counter() - t0) * 1000, 1),
                            note=None if n else "returned no rows"))
        except Exception as exc:
            out.append(dict(b, rows=None, ms=None,
                            note=f"failed: {str(exc).splitlines()[0][:160]}"))
    return out


def load_snapshot(path: Path = SNAPSHOT):
    """Concepts and relations as read from Neo4j, for a view that cannot connect."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("source") != "neo4j":
        raise ValueError(f"{path} is not a Neo4j snapshot")
    return d["concepts"], d["relations"], d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolt", default=os.environ.get("NEO4J_URI",
                                                     "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD",
                                                         "jeju-ontology-2026"))
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--version", default="1.4.0")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the comparison against the file store")
    ap.add_argument("--user-hint", default="tahirahmad@203.253.207.64",
                    help="what the SSH tunnel line should say")
    ap.add_argument("--out", default=str(SNAPSHOT))
    a = ap.parse_args()

    d = export(a.bolt, a.user, a.password, a.database, a.version,
               do_verify=not a.no_verify, user_hint=a.user_hint)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    script = DELIV / "whole_ontology.cypher"
    script.write_text(browser_script_text(d["meta"], d["mapping"],
                                          d["browser_script"]),
                      encoding="utf-8")

    m = d["meta"]
    print(f"{m['engine']} {m['edition']} at {m['bolt']} (db {m['database']})")
    mp = d["mapping"]
    print(f"schema in the database: {mp['name']} - "
          f"{len(mp['labels_in_use'])} label(s), "
          f"{len(mp['relationship_types_in_use'])} relationship type(s)")
    print(f"read over Cypher: {len(d['concepts']):,} concepts, "
          f"{len(d['relations']):,} relations")
    if d["nodes_not_returned_by_the_concept_query"]:
        print(f"  warning: {d['nodes_not_returned_by_the_concept_query']} nodes "
              f"carry no :Concept label and were not exported")
    if d["relationships_not_returned_by_the_relation_query"]:
        print(f"  warning: "
              f"{d['relationships_not_returned_by_the_relation_query']} "
              f"relationships fall outside Concept->Concept")
    v = d.get("verification")
    if v:
        if v["exact"]:
            print(f"verified exact against the v{v['file_store_version']} "
                  f"file store: same {v['file_store_concepts']:,} names, same "
                  f"{v['file_store_relations']:,} triples, same classes")
        else:
            print("MISMATCH against the file store:")
            for k in ("missing_concepts", "extra_concepts", "missing_relations",
                      "extra_relations", "concepts_with_a_different_class"):
                if v[k]:
                    print(f"  {k}: {v[k]}")
    print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB)")
    print(f"wrote {script} ({len(d['browser_script'])} Browser queries for the "
          f"{d['mapping']['name']} mapping)")
    flagged = [b for b in d["browser_script"] if b.get("note")
               and b["note"] != "Browser client command, not Cypher"]
    for b in flagged:
        print(f"  note: \"{b['title']}\" {b['note']}")


if __name__ == "__main__":
    main()
