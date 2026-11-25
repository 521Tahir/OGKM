#!/usr/bin/env python3
"""Build an interactive HTML viewer for the Jeju Tourism Ontology.

Parses an OWL/RDF ontology file (Turtle by default) with rdflib and writes
`ontology_viewer/data.js`, which `index.html` loads. Then serve the folder:

    python ontology_viewer/build_viewer.py            # build data.js
    python ontology_viewer/build_viewer.py --serve    # build and serve on :8790

Options:
    --ttl PATH    ontology file (default: wbs/deliverables/task2_formalization/jeju_tourism_ontology.ttl)
    --port N      port for --serve (default 8765)
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_TTL = PROJECT / "wbs/deliverables/task2_formalization/jeju_tourism_ontology.ttl"

META_TYPES = {
    OWL.Class, OWL.NamedIndividual, OWL.Ontology, OWL.ObjectProperty,
    OWL.DatatypeProperty, OWL.AnnotationProperty, OWL.SymmetricProperty,
    OWL.TransitiveProperty, OWL.FunctionalProperty, OWL.InverseFunctionalProperty,
    OWL.ReflexiveProperty, OWL.IrreflexiveProperty, OWL.AsymmetricProperty,
    RDFS.Class, RDF.Property,
}
PROP_KIND = {
    OWL.ObjectProperty: "ObjectProperty",
    OWL.DatatypeProperty: "DatatypeProperty",
    OWL.AnnotationProperty: "AnnotationProperty",
    RDF.Property: "rdf:Property",
}
CHARACTERISTICS = {
    OWL.SymmetricProperty: "Symmetric", OWL.TransitiveProperty: "Transitive",
    OWL.FunctionalProperty: "Functional", OWL.InverseFunctionalProperty: "InverseFunctional",
    OWL.ReflexiveProperty: "Reflexive", OWL.IrreflexiveProperty: "Irreflexive",
    OWL.AsymmetricProperty: "Asymmetric",
}


def build(ttl: Path) -> dict:
    g = Graph()
    fmt = "turtle" if ttl.suffix == ".ttl" else None
    g.parse(ttl, format=fmt)
    nm = g.namespace_manager

    def q(term) -> str:
        if isinstance(term, URIRef):
            try:
                return nm.normalizeUri(term)
            except Exception:
                return str(term)
        if isinstance(term, BNode):
            return "_:" + str(term)
        return str(term)

    def lit(o: Literal):
        return {"v": str(o), "dt": q(o.datatype) if o.datatype else None, "lang": o.language}

    def label(s) -> str:
        for p in (RDFS.label, URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")):
            v = g.value(s, p)
            if v is not None:
                return str(v)
        return q(s).split(":", 1)[-1]

    # ---------- ontology header ----------
    onto = []
    for s in g.subjects(RDF.type, OWL.Ontology):
        onto.append({"iri": str(s), "props": [[q(p), q(o) if not isinstance(o, Literal) else str(o)]
                                               for p, o in g.predicate_objects(s)]})

    # ---------- classes ----------
    class_nodes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    for s, o in g.subject_objects(RDFS.subClassOf):
        class_nodes.update([s, o])
    for o in g.objects(None, RDF.type):
        if o not in META_TYPES and isinstance(o, URIRef):
            class_nodes.add(o)
    class_nodes = {c for c in class_nodes if isinstance(c, URIRef)}

    # ---------- individuals ----------
    indiv_nodes = set(g.subjects(RDF.type, OWL.NamedIndividual))
    for s, o in g.subject_objects(RDF.type):
        if o in class_nodes and isinstance(s, URIRef):
            indiv_nodes.add(s)

    def asserted_types(s):
        return [o for o in g.objects(s, RDF.type) if o not in META_TYPES]

    # ---------- properties ----------
    def usage_once(p):
        return sum(1 for _ in g.triples((None, p, None))) <= 1

    prop_nodes: dict = {}
    for t, kind in PROP_KIND.items():
        for s in g.subjects(RDF.type, t):
            prop_nodes.setdefault(s, {"kind": kind, "declared": True})
    builtin = {RDF.type, RDFS.subClassOf, OWL.versionInfo}
    header_preds = {p for o in g.subjects(RDF.type, OWL.Ontology) for p in g.predicates(o)}
    for p in set(g.predicates()):
        if p in builtin or (p in header_preds and usage_once(p)):
            continue
        if p not in prop_nodes:
            prop_nodes[p] = {"kind": "Annotation / data attribute" if all(
                isinstance(o, Literal) for o in g.objects(None, p)) else "Resource link (undeclared)",
                "declared": False}

    usage = Counter(p for _, p, _ in g)
    dom_obs: dict = defaultdict(Counter)
    rng_obs: dict = defaultdict(Counter)
    schema_edges: Counter = Counter()  # (srcClass, prop, dstClass) -> count
    for s, p, o in g:
        if p == RDF.type:
            continue
        st = asserted_types(s) or ([OWL.Class] if s in class_nodes else [])
        for t in st:
            dom_obs[p][q(t)] += 1
        if isinstance(o, Literal):
            rng_obs[p][q(o.datatype) if o.datatype else "xsd:string"] += 1
        else:
            ot = asserted_types(o) or ([OWL.Class] if o in class_nodes else [])
            for t in ot:
                rng_obs[p][q(t)] += 1
            if prop_nodes.get(p, {}).get("kind") == "ObjectProperty":
                for a in st:
                    for b in ot:
                        if a in class_nodes and b in class_nodes:
                            schema_edges[(q(a), q(p), q(b))] += 1

    props = []
    for p, info in prop_nodes.items():
        props.append({
            "id": q(p), "iri": str(p), "label": str(g.value(p, RDFS.label) or ""),
            "kind": info["kind"], "declared": info["declared"],
            "characteristics": sorted(CHARACTERISTICS[t] for t in g.objects(p, RDF.type) if t in CHARACTERISTICS),
            "domain": [q(o) for o in g.objects(p, RDFS.domain)],
            "range": [q(o) for o in g.objects(p, RDFS.range)],
            "inverseOf": [q(o) for o in g.objects(p, OWL.inverseOf)] + [q(s) for s in g.subjects(OWL.inverseOf, p)],
            "subPropertyOf": [q(o) for o in g.objects(p, RDFS.subPropertyOf)],
            "comment": str(g.value(p, RDFS.comment) or ""),
            "usage": usage[p],
            "observedDomain": dom_obs[p].most_common(),
            "observedRange": rng_obs[p].most_common(),
        })
    props.sort(key=lambda x: (x["kind"], -x["usage"]))

    # ---------- per-resource details ----------
    def details(s):
        out = []
        for p, o in sorted(g.predicate_objects(s), key=lambda po: (q(po[0]), str(po[1]))):
            if isinstance(o, Literal):
                out.append([q(p), lit(o)])
            else:
                out.append([q(p), q(o)])
        return out

    def incoming(s):
        return sorted([q(p), q(x)] for x, p in g.subject_predicates(s) if p != RDF.type)

    direct_inst: Counter = Counter()
    for i in indiv_nodes:
        for t in asserted_types(i):
            direct_inst[q(t)] += 1

    classes = []
    for c in class_nodes:
        cid = q(c)
        classes.append({
            "id": cid, "iri": str(c), "label": label(c),
            "declared": (c, RDF.type, OWL.Class) in g,
            "super": [q(o) for o in g.objects(c, RDFS.subClassOf)],
            "directInstances": direct_inst[cid],
            "props": details(c), "incoming": incoming(c),
            "equivalent": [q(o) for o in g.objects(c, OWL.equivalentClass)],
            "disjoint": [q(o) for o in g.objects(c, OWL.disjointWith)],
        })
    classes.sort(key=lambda x: x["label"].lower())

    individuals = []
    for i in indiv_nodes:
        individuals.append({
            "id": q(i), "iri": str(i), "label": label(i),
            "types": [q(t) for t in asserted_types(i)],
            "isClass": i in class_nodes,
            "props": details(i), "incoming": incoming(i),
        })
    individuals.sort(key=lambda x: x["label"].lower())

    # instance-level edges (resource-valued, non-rdf:type)
    links = Counter()
    for s, p, o in g:
        if p in (RDF.type,) or isinstance(o, Literal) or isinstance(o, BNode):
            continue
        if s in indiv_nodes and (o in indiv_nodes or o in class_nodes):
            links[(q(s), q(p), q(o))] += 1

    return {
        "source": str(ttl.relative_to(PROJECT)) if ttl.is_relative_to(PROJECT) else str(ttl),
        "ontology": onto,
        "namespaces": sorted([pfx, str(ns)] for pfx, ns in nm.namespaces() if pfx),
        "stats": {
            "triples": len(g), "classes": len(classes), "individuals": len(individuals),
            "objectProperties": sum(p["kind"] == "ObjectProperty" for p in props),
            "datatypeProperties": sum(p["kind"] == "DatatypeProperty" for p in props),
            "annotationProperties": sum(p["kind"] == "AnnotationProperty" for p in props),
            "otherProperties": sum(not p["declared"] for p in props),
        },
        "classes": classes,
        "properties": props,
        "individuals": individuals,
        "schemaEdges": [[a, p, b, n] for (a, p, b), n in schema_edges.most_common()],
        "links": [[a, p, b] for (a, p, b) in links],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttl", type=Path, default=DEFAULT_TTL)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 = reachable on the network)")
    args = ap.parse_args()

    data = build(args.ttl.resolve())
    out = HERE / "data.js"
    out.write_text("window.ONTOLOGY = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    s = data["stats"]
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB): {s['triples']} triples, {s['classes']} classes, "
          f"{s['objectProperties']} object props, {s['individuals']} individuals")

    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer((args.host, args.port), handler) as httpd:
            print(f"Serving ontology viewer at http://{args.host}:{args.port}/  (Ctrl+C to stop)")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
