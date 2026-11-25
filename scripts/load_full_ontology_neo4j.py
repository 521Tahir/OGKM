#!/usr/bin/env python3
"""Load the COMPLETE Jeju Tourism Ontology (OWL/Turtle) into Neo4j.

Every IRI in the ontology becomes one node, so nothing is dropped:

  Schema layer
    (:Ontology)                         title, version, creator, iri
    (:Class)                            the 33 OWL classes (rdfs:label, confidence, provenance)
    (:Class)-[:SUBCLASS_OF]->(:Class)   rdfs:subClassOf hierarchy (root: TourismEntity)
    (:Property:ObjectProperty)          the 19 object properties (label, symmetric, usage count)
    (:Property:AnnotationProperty)      literal attributes (confidence, attr_lat, skos labels, ...)
    (:Class)-[:SCHEMA_LINK {property, count}]->(:Class)
                                        observed domain -> range of every object property

  Instance layer
    (:Individual:<ClassName>)           the 1,564 named individuals, all literal values as properties
    (:Individual)-[:INSTANCE_OF]->(:Class)              rdf:type
    (:Individual)-[:LOCATED_IN|NEAR|PART_OF|...]->(...) every object-property assertion,
                                        one relationship type per predicate

Usage:
    python scripts/load_full_ontology_neo4j.py              # clear DB and load
    python scripts/load_full_ontology_neo4j.py --no-clear   # merge into existing graph
    python scripts/load_full_ontology_neo4j.py --ttl other.ttl

Connection comes from .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD).
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from neo4j import GraphDatabase
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTL = ROOT / "wbs/deliverables/task2_formalization/jeju_tourism_ontology.ttl"
SKOS = "http://www.w3.org/2004/02/skos/core#"
META = {OWL.Class, OWL.NamedIndividual, OWL.Ontology, OWL.ObjectProperty, OWL.DatatypeProperty,
        OWL.AnnotationProperty, OWL.SymmetricProperty, OWL.TransitiveProperty, OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty, RDFS.Class, RDF.Property}
SAFE = re.compile(r"[^A-Za-z0-9_]")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def local(iri) -> str:
    s = str(iri)
    return re.split(r"[#/]", s)[-1] or s


def label_name(s: str) -> str:
    s = SAFE.sub("_", s)
    return s if s and not s[0].isdigit() else "C_" + s


def rel_name(pred: URIRef) -> str:
    return label_name(local(pred)).upper()


def pyval(o: Literal):
    v = o.toPython()
    return v if isinstance(v, (int, float, bool, str)) else str(v)


def build(ttl: Path):
    g = Graph()
    g.parse(ttl, format="turtle")

    classes = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    for s, o in g.subject_objects(RDFS.subClassOf):
        classes.update(x for x in (s, o) if isinstance(x, URIRef))
    for o in g.objects(None, RDF.type):
        if isinstance(o, URIRef) and o not in META:
            classes.add(o)

    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    symmetric = set(g.subjects(RDF.type, OWL.SymmetricProperty))

    # ---- nodes: one per subject/object IRI ----
    nodes: dict[URIRef, dict] = defaultdict(lambda: {"props": {}, "labels": set()})
    onto = next(g.subjects(RDF.type, OWL.Ontology), None)

    for s in set(g.subjects()) | {o for o in g.objects() if isinstance(o, URIRef)}:
        if not isinstance(s, URIRef) or s in META or s == onto:
            continue
        if s in obj_props or (s, RDF.type, OWL.ObjectProperty) in g:
            continue  # property nodes are built separately
        if str(s).startswith(("http://www.w3.org/", "http://purl.org/", "http://www.w3.org/ns/prov")):
            continue
        n = nodes[s]
        n["props"].update(iri=str(s), id=local(s))
        types = [t for t in g.objects(s, RDF.type) if t not in META]
        if s in classes:
            n["labels"].add("Class")
        if (s, RDF.type, OWL.NamedIndividual) in g or types:
            n["labels"].add("Individual")
            n["labels"].update(label_name(local(t)) for t in types)
            n["props"]["types"] = sorted(local(t) for t in types)
        if not n["labels"]:
            n["labels"].add("Resource")
        multi = defaultdict(list)
        for p, o in g.predicate_objects(s):
            if isinstance(o, Literal):
                multi[local(p)].append(pyval(o))
        for k, vals in multi.items():
            n["props"][k] = vals[0] if len(vals) == 1 else vals
        n["props"]["name"] = str(g.value(s, RDFS.label) or g.value(s, URIRef(SKOS + "prefLabel")) or local(s))
        if "wasGeneratedBy" in n["props"]:
            n["props"]["generatedBy"] = n["props"].pop("wasGeneratedBy")
        if "attr_lat" in n["props"]:
            n["props"]["latitude"] = n["props"]["attr_lat"]
            n["props"]["longitude"] = n["props"].get("attr_lon")

    # ---- edges ----
    usage = Counter()
    edges = defaultdict(list)  # reltype -> [(s,o,predicate_iri)]
    for s, p, o in g:
        if isinstance(o, Literal) or s not in nodes or o not in nodes:
            continue
        if p == RDF.type:
            if o in classes:
                edges["INSTANCE_OF"].append((str(s), str(o)))
        elif p == RDFS.subClassOf:
            edges["SUBCLASS_OF"].append((str(s), str(o)))
        else:
            usage[p] += 1
            edges[rel_name(p)].append((str(s), str(o)))

    # observed schema links: class -> class per object property
    def types_of(x):
        return [t for t in g.objects(x, RDF.type) if t in classes]
    schema = Counter()
    for s, p, o in g:
        if p in obj_props and not isinstance(o, Literal):
            for a in types_of(s):
                for b in (types_of(o) or ([o] if o in classes else [])):
                    schema[(str(a), local(p), str(b))] += 1

    # ---- property nodes ----
    lit_usage = Counter(p for _, p, o in g if isinstance(o, Literal))
    props = []
    for p in obj_props:
        props.append({"iri": str(p), "name": local(p), "label": str(g.value(p, RDFS.label) or local(p)),
                      "relType": rel_name(p), "symmetric": p in symmetric, "usage": usage[p],
                      "kind": "ObjectProperty"})
    for p, n in lit_usage.items():
        if p in (RDFS.label, OWL.versionInfo) or str(p).startswith("http://purl.org/dc/terms/"):
            continue
        props.append({"iri": str(p), "name": local(p), "label": local(p), "usage": n, "kind": "AnnotationProperty"})

    meta = {"iri": str(onto)} if onto else {}
    if onto:
        for p, o in g.predicate_objects(onto):
            if isinstance(o, Literal):
                meta[local(p)] = str(o)
    meta["triples"] = len(g)
    meta["source"] = str(ttl.relative_to(ROOT)) if ttl.is_relative_to(ROOT) else str(ttl)
    return meta, nodes, edges, props, schema


def chunks(xs, n=1000):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttl", type=Path, default=DEFAULT_TTL)
    ap.add_argument("--no-clear", action="store_true", help="do not wipe the database first")
    args = ap.parse_args()

    t0 = time.perf_counter()
    meta, nodes, edges, props, schema = build(args.ttl.resolve())
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    drv = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "")))
    drv.verify_connectivity()

    with drv.session() as s:
        if not args.no_clear:
            s.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
            for r in s.run("SHOW CONSTRAINTS YIELD name").data():
                s.run(f"DROP CONSTRAINT `{r['name']}` IF EXISTS").consume()
            for r in s.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").data():
                s.run(f"DROP INDEX `{r['name']}` IF EXISTS").consume()
        s.run("CREATE CONSTRAINT onto_iri IF NOT EXISTS FOR (n:OntoNode) REQUIRE n.iri IS UNIQUE").consume()
        s.run("CREATE INDEX onto_name IF NOT EXISTS FOR (n:OntoNode) ON (n.name)").consume()
        s.run("CREATE FULLTEXT INDEX onto_search IF NOT EXISTS FOR (n:OntoNode) ON EACH [n.name, n.id]").consume()

        s.run("MERGE (o:Ontology {iri:$m.iri}) SET o += $m, o.name = coalesce($m.title, 'Ontology')", m=meta).consume()

        # nodes grouped by label set (labels can't be parameterised)
        groups = defaultdict(list)
        for n in nodes.values():
            groups[tuple(sorted(n["labels"]))].append(n["props"])
        for labels, rows in groups.items():
            lab = ":".join(f"`{l}`" for l in ("OntoNode",) + labels)
            for batch in chunks(rows):
                s.run(f"UNWIND $rows AS r MERGE (n:OntoNode {{iri:r.iri}}) SET n += r SET n{lab[len('`OntoNode`'):]}",
                      rows=batch).consume()

        for rt, pairs in edges.items():
            for batch in chunks([{"s": a, "o": b} for a, b in pairs], 2000):
                s.run(f"UNWIND $rows AS r MATCH (a:OntoNode {{iri:r.s}}), (b:OntoNode {{iri:r.o}}) "
                      f"MERGE (a)-[:`{rt}`]->(b)", rows=batch).consume()

        s.run("UNWIND $rows AS r MERGE (p:OntoNode {iri:r.iri}) SET p += r SET p:Property "
              "WITH p, r CALL { WITH p, r WITH p WHERE r.kind='ObjectProperty' SET p:ObjectProperty } "
              "CALL { WITH p, r WITH p WHERE r.kind='AnnotationProperty' SET p:AnnotationProperty } "
              "WITH p MATCH (o:Ontology) MERGE (o)-[:DECLARES]->(p)", rows=props).consume()
        s.run("MATCH (o:Ontology), (c:Class) WHERE NOT (c)-[:SUBCLASS_OF]->() MERGE (o)-[:HAS_ROOT_CLASS]->(c)").consume()
        s.run("UNWIND $rows AS r MATCH (a:Class {iri:r.a}), (b:Class {iri:r.b}) "
              "MERGE (a)-[l:SCHEMA_LINK {property:r.p}]->(b) SET l.count = r.n",
              rows=[{"a": a, "p": p, "b": b, "n": n} for (a, p, b), n in schema.items()]).consume()
        s.run("MATCH (c:Class) OPTIONAL MATCH (i:Individual)-[:INSTANCE_OF]->(c) "
              "WITH c, count(i) AS n SET c.directInstances = n").consume()

        nn = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        nr = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        labs = s.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC").data()
        rts = s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC").data()
    drv.close()

    print(f"Loaded {meta.get('title')} v{meta.get('versionInfo')} from {meta['source']} in {time.perf_counter()-t0:.1f}s")
    print(f"  nodes: {nn:,}   relationships: {nr:,}")
    print("  labels:", ", ".join(f"{r['l']}={r['c']}" for r in labs))
    print("  rel types:", ", ".join(f"{r['t']}={r['c']}" for r in rts))


if __name__ == "__main__":
    main()
