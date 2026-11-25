"""
WBS Task 2.2.2 - Ontology validation engine.

Four independent layers, each producing machine-readable findings with a repair
suggestion attached. Nothing is auto-repaired: the engine reports, the governance
workflow (Task 3.1.2) decides.

  L1 STRUCTURAL   well-formedness of the JSON records themselves - missing names,
                  empty predicates, relations pointing at absent concepts,
                  duplicate assertions, self-loops.
  L2 LOGICAL      ontology-level consistency that OWL alone will not catch here:
                  class-cycle detection, multi-class instances with incompatible
                  parents, symmetric-property asymmetry, and type-plausibility of
                  (subject, predicate, object) against the class tree.
  L3 SHACL        a real shapes graph executed by pySHACL against the OWL/Turtle
                  serialisation - cardinality, datatype, label and class-membership
                  constraints.
  L4 OWL          OWL-RL closure via owlrl, then a consistency probe for
                  owl:Nothing membership and contradictory class assertions.

Quality metrics are computed alongside (richness, density, connectivity,
provenance coverage), following the standard OntoQA-style descriptors.

  python -m wbs.task2_formalization.validate_ontology --version 1.3.0-norm
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NS = "http://jejunu.ac.kr/ontology/jeju-tourism#"

# Predicates whose subject/object classes are constrained by the domain.
# Only rules we can defend from the schema are listed; everything else is left
# unconstrained rather than inventing a restriction the data never promised.
TYPE_RULES = {
    "located_in": {"object_in": {"Region", "City", "Island", "Destination"}},
    "part_of":    {"object_in": {"Region", "City", "Island", "Destination",
                                 "NaturalAttraction", "Attraction"}},
    "served_at":  {"subject_in": {"Cuisine"}},
    "has_activity": {"object_in": {"Activity"}},
    "reachable_by": {"object_in": {"TransportMode", "Route", "Transportation"}},
    "best_for":   {"object_in": {"TravelerType", "Season", "WeatherCondition"}},
    "has_seasonality": {"object_in": {"Season", "WeatherCondition"}},
}
SYMMETRIC = {"near", "co_occurs_with", "connected_to"}


def key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


class Finding(dict):
    def __init__(self, layer, severity, code, message, subject=None, repair=None):
        super().__init__(layer=layer, severity=severity, code=code, message=message,
                         subject=subject, repair_suggestion=repair)


def load(version):
    d = ROOT / "artifacts" / "ontologies" / version
    rd = lambda f: json.loads((d / f).read_text(encoding="utf-8")) if (d / f).exists() else []
    return rd("concepts.json"), rd("relations.json"), rd("hierarchy.json"), d


# ── L1 structural ─────────────────────────────────────────────────────────
def l1_structural(concepts, relations):
    f = []
    names = {key(c["name"]) for c in concepts if isinstance(c, dict) and c.get("name")}
    for i, c in enumerate(concepts):
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            f.append(Finding("L1", "error", "CONCEPT_NO_NAME",
                             f"concept at index {i} has no usable name", repair="drop the record"))
            continue
        if not c.get("concept_class"):
            f.append(Finding("L1", "warning", "CONCEPT_UNTYPED",
                             f"'{c['name']}' has no concept_class", c["name"],
                             "assign a class from the schema or route to expert review"))
    seen = set()
    for r in relations:
        if not isinstance(r, dict):
            continue
        s, p, o = r.get("subject"), r.get("predicate"), r.get("object")
        if not (s and p and o):
            f.append(Finding("L1", "error", "RELATION_INCOMPLETE",
                             f"incomplete triple {s!r} {p!r} {o!r}", repair="drop the triple"))
            continue
        if key(s) == key(o):
            f.append(Finding("L1", "error", "SELF_LOOP", f"'{s}' {p} itself", s,
                             "drop the triple"))
        sig = (key(s), p, key(o))
        if sig in seen:
            f.append(Finding("L1", "warning", "DUPLICATE_TRIPLE",
                             f"duplicate: {s} {p} {o}", s, "deduplicate"))
        seen.add(sig)
        for end, label in ((s, "subject"), (o, "object")):
            if key(end) not in names:
                f.append(Finding("L1", "warning", "DANGLING_ENDPOINT",
                                 f"{label} '{end}' of '{s} {p} {o}' is not a declared concept",
                                 end, "declare the concept or drop the triple"))
    return f


# ── L2 logical ────────────────────────────────────────────────────────────
def l2_logical(concepts, relations, hierarchy):
    f = []
    cls_of = {}
    for h in hierarchy:
        if isinstance(h, dict) and h.get("concept"):
            cls_of[key(h["concept"])] = h.get("concept_class")
    for c in concepts:
        if isinstance(c, dict) and c.get("name") and c.get("concept_class"):
            cls_of.setdefault(key(c["name"]), c["concept_class"])

    # class cycles
    parent = defaultdict(set)
    for h in hierarchy:
        path = (h or {}).get("hierarchy_path") or []
        for a, b in zip(path, path[1:]):
            parent[b].add(a)
    for start in list(parent):
        seen, dq = set(), deque([start])
        while dq:
            n = dq.popleft()
            for p in parent.get(n, ()):
                if p == start:
                    f.append(Finding("L2", "error", "CLASS_CYCLE",
                                     f"class '{start}' is its own ancestor via '{n}'", start,
                                     "break the cycle in hierarchy_path"))
                    dq.clear(); break
                if p not in seen:
                    seen.add(p); dq.append(p)

    # instances declared under conflicting classes
    multi = defaultdict(set)
    for h in hierarchy:
        if isinstance(h, dict) and h.get("concept") and h.get("concept_class"):
            multi[key(h["concept"])].add(h["concept_class"])
    for k, classes in multi.items():
        if len(classes) > 1:
            f.append(Finding("L2", "warning", "MULTI_CLASS_INSTANCE",
                             f"'{k}' declared as {sorted(classes)}", k,
                             "pick the most specific class or declare them compatible"))

    # symmetric properties asserted one-way only
    idx = {(key(r.get("subject")), r.get("predicate"), key(r.get("object")))
           for r in relations if isinstance(r, dict) and r.get("subject") and r.get("object")}
    for r in relations:
        if not isinstance(r, dict):
            continue
        p = r.get("predicate")
        if p in SYMMETRIC:
            if (key(r["object"]), p, key(r["subject"])) not in idx:
                f.append(Finding("L2", "info", "SYMMETRY_NOT_MATERIALISED",
                                 f"'{r['subject']}' {p} '{r['object']}' has no inverse assertion",
                                 r["subject"],
                                 "materialise the inverse, or rely on the OWL SymmetricProperty"))

    # type plausibility
    for r in relations:
        if not isinstance(r, dict):
            continue
        p = r.get("predicate")
        rule = TYPE_RULES.get(p)
        if not rule:
            continue
        sc, oc = cls_of.get(key(r.get("subject"))), cls_of.get(key(r.get("object")))
        if "subject_in" in rule and sc and sc not in rule["subject_in"]:
            f.append(Finding("L2", "warning", "TYPE_VIOLATION_SUBJECT",
                             f"'{r['subject']}' ({sc}) {p} '{r['object']}' - subject class not in "
                             f"{sorted(rule['subject_in'])}", r["subject"],
                             "retype the subject or drop the triple"))
        if "object_in" in rule and oc and oc not in rule["object_in"]:
            f.append(Finding("L2", "warning", "TYPE_VIOLATION_OBJECT",
                             f"'{r['subject']}' {p} '{r['object']}' ({oc}) - object class not in "
                             f"{sorted(rule['object_in'])}", r["object"],
                             "retype the object or drop the triple"))
    return f


# ── L3 SHACL ──────────────────────────────────────────────────────────────
SHAPES = f"""
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix jeju: <{NS}> .

jeju:IndividualShape a sh:NodeShape ;
    sh:targetClass owl:NamedIndividual ;
    sh:property [ sh:path rdfs:label ; sh:minCount 1 ; sh:maxCount 1 ;
                  sh:datatype xsd:string ;
                  sh:message "every individual needs exactly one rdfs:label" ] ;
    sh:property [ sh:path skos:prefLabel ; sh:minCount 1 ;
                  sh:message "every individual needs a skos:prefLabel" ] ;
    sh:property [ sh:path jeju:confidence ; sh:maxCount 1 ;
                  sh:datatype xsd:double ;
                  sh:minInclusive 0.0 ; sh:maxInclusive 1.0 ;
                  sh:message "confidence must be a single double in [0,1]" ] .

jeju:ClassShape a sh:NodeShape ;
    sh:targetClass owl:Class ;
    sh:property [ sh:path rdfs:label ; sh:minCount 1 ;
                  sh:message "every class needs an rdfs:label" ] .

jeju:LocatedInShape a sh:NodeShape ;
    sh:targetSubjectsOf jeju:located_in ;
    sh:property [ sh:path jeju:located_in ; sh:nodeKind sh:IRI ;
                  sh:message "located_in must point at a resource, not a literal" ] .
"""


def l3_shacl(ttl_path: Path, limit: int = 400):
    f = []
    try:
        from pyshacl import validate as shacl_validate
        from rdflib import Graph
    except Exception as e:                                    # pragma: no cover
        return [Finding("L3", "info", "SHACL_UNAVAILABLE", f"pySHACL not usable: {e}")], None
    data = Graph(); data.parse(str(ttl_path), format="turtle")
    shapes = Graph(); shapes.parse(data=SHAPES, format="turtle")
    conforms, results_graph, _ = shacl_validate(
        data, shacl_graph=shapes, inference="none", abort_on_first=False,
        allow_infos=True, allow_warnings=True, meta_shacl=False, advanced=False)
    from rdflib.namespace import Namespace
    SH = Namespace("http://www.w3.org/ns/shacl#")
    n = 0
    for res in results_graph.subjects(predicate=SH.resultSeverity):
        msg = next(results_graph.objects(res, SH.resultMessage), "")
        focus = next(results_graph.objects(res, SH.focusNode), "")
        sev = str(next(results_graph.objects(res, SH.resultSeverity), "")).split("#")[-1]
        n += 1
        if n <= limit:
            f.append(Finding("L3", "error" if sev == "Violation" else "warning",
                             "SHACL_" + sev.upper(), str(msg), str(focus).split("#")[-1],
                             "fix the shape violation or relax the shape"))
    return f, {"conforms": bool(conforms), "total_results": n, "reported": min(n, limit)}


# ── L4 OWL reasoning ──────────────────────────────────────────────────────
def l4_owl(ttl_path: Path):
    f, stats = [], {}
    try:
        import owlrl
        from rdflib import Graph, OWL, RDF
    except Exception as e:                                    # pragma: no cover
        return [Finding("L4", "info", "OWLRL_UNAVAILABLE", str(e))], stats
    g = Graph(); g.parse(str(ttl_path), format="turtle")
    before = len(g)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics, rdfs_closure=True,
                           axiomatic_triples=False, datatype_axioms=False).expand(g)
    stats = {"triples_before_closure": before, "triples_after_closure": len(g),
             "entailed": len(g) - before}
    nothing = list(g.subjects(RDF.type, OWL.Nothing))
    if nothing:
        for s in nothing[:50]:
            f.append(Finding("L4", "error", "OWL_INCONSISTENT",
                             f"{str(s).split('#')[-1]} entailed as owl:Nothing",
                             str(s).split("#")[-1], "remove the contradictory assertion"))
    else:
        f.append(Finding("L4", "info", "OWL_CONSISTENT",
                         "no individual entailed as owl:Nothing after OWL-RL closure"))
    return f, stats


# ── quality metrics ───────────────────────────────────────────────────────
def quality_metrics(concepts, relations, hierarchy):
    n_c, n_r = len(concepts), len(relations)
    classes = {h.get("concept_class") for h in hierarchy if isinstance(h, dict)} - {None}
    preds = Counter(r["predicate"] for r in relations if isinstance(r, dict) and r.get("predicate"))
    deg = Counter()
    for r in relations:
        if isinstance(r, dict) and r.get("subject") and r.get("object"):
            deg[key(r["subject"])] += 1; deg[key(r["object"])] += 1
    names = {key(c["name"]) for c in concepts if isinstance(c, dict) and c.get("name")}
    isolated = len(names - set(deg))
    depths = [len(h.get("hierarchy_path") or []) for h in hierarchy if isinstance(h, dict)]
    with_prov = sum(1 for c in concepts if isinstance(c, dict)
                    and isinstance(c.get("provenance"), dict)
                    and c["provenance"].get("confidence") is not None)
    with_alias = sum(1 for c in concepts if isinstance(c, dict) and (c.get("aliases") or []))
    return {
        "concepts": n_c, "relations": n_r, "classes": len(classes),
        "distinct_predicates": len(preds),
        # every ratio below is reported as null rather than 0.0 when its
        # denominator is empty - an ontology with no classes has no measured
        # richness, which is not the same claim as no richness (R15)
        "relationship_richness": rate(len(preds), classes),
        "attribute_richness": rate(
            sum(len((c.get("attributes") or {})) for c in concepts if isinstance(c, dict)),
            n_c),
        "inheritance_richness": rate(len(hierarchy), classes),
        "graph_density": rate(n_r, n_c * (n_c - 1), 8),
        "average_degree": rate(sum(deg.values()), n_c),
        "max_degree": max(deg.values()) if deg else 0,
        "isolated_concepts": isolated,
        "isolated_fraction": rate(isolated, n_c),
        "mean_hierarchy_depth": rate(sum(depths), depths),
        "max_hierarchy_depth": max(depths) if depths else 0,
        "provenance_coverage": rate(with_prov, n_c),
        "alias_coverage": rate(with_alias, n_c),
        "top_predicates": preds.most_common(10),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.3.0-norm")
    # Validating a non-published version used to overwrite the published
    # report and quality metrics, because the output names carry no version.
    # A version other than the default must now name its own outputs.
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output files, e.g. __1.5.0")
    a = ap.parse_args()

    concepts, relations, hierarchy, d = load(a.version)
    print(f"validating v{a.version}: {len(concepts):,} concepts, {len(relations):,} relations")

    findings = []
    print("  L1 structural …", flush=True)
    findings += l1_structural(concepts, relations)
    print("  L2 logical …", flush=True)
    findings += l2_logical(concepts, relations, hierarchy)

    ttl = ROOT / "wbs" / "deliverables" / "task2_formalization" / "jeju_tourism_ontology.ttl"
    shacl_stats = owl_stats = None
    if ttl.exists():
        print("  L3 SHACL …", flush=True)
        f3, shacl_stats = l3_shacl(ttl)
        findings += f3
        print("  L4 OWL-RL closure …", flush=True)
        f4, owl_stats = l4_owl(ttl)
        findings += f4
    else:
        findings.append(Finding("L3", "info", "NO_TTL", "run normalize_and_formalize first"))

    metrics = quality_metrics(concepts, relations, hierarchy)
    sev = Counter(f["severity"] for f in findings)
    codes = Counter(f["code"] for f in findings)

    D = ROOT / "wbs" / "deliverables" / "task2_formalization"
    D.mkdir(parents=True, exist_ok=True)
    tag = a.out_tag
    if a.version != ap.get_default("version") and not tag:
        sys.exit(f"refusing to overwrite the published report with {a.version}: "
                 f"pass --out-tag (e.g. --out-tag __{a.version})")
    (D / f"validation_report{tag}.json").write_text(json.dumps(
        {"version": a.version, "severity_counts": dict(sev), "code_counts": dict(codes),
         "shacl": shacl_stats, "owl": owl_stats, "quality_metrics": metrics,
         "findings": findings}, indent=2, ensure_ascii=False), encoding="utf-8")
    (D / f"ontology_quality_metrics{tag}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    md = ["# Ontology Validation Report", "", f"**Ontology version:** {a.version}", "",
          "## Findings by severity", "", "| Severity | Count |", "|---|---|"]
    for k in ("error", "warning", "info"):
        md.append(f"| {k} | {sev.get(k, 0):,} |")
    md += ["", "## Findings by code", "", "| Layer | Code | Count | Example repair |", "|---|---|---|---|"]
    ex = {}
    for f in findings:
        ex.setdefault(f["code"], f)
    for code, n in codes.most_common():
        e = ex[code]
        md.append(f"| {e['layer']} | `{code}` | {n:,} | {e['repair_suggestion'] or '—'} |")
    if shacl_stats:
        md += ["", "## SHACL", "",
               f"- conforms: **{shacl_stats['conforms']}**",
               f"- results: {shacl_stats['total_results']:,} "
               f"(first {shacl_stats['reported']:,} itemised in the JSON report)"]
    if owl_stats:
        md += ["", "## OWL-RL closure", "",
               f"- triples before: {owl_stats['triples_before_closure']:,}",
               f"- triples after: {owl_stats['triples_after_closure']:,}",
               f"- entailed: {owl_stats['entailed']:,}"]
    md += ["", "## Quality metrics", "", "| Metric | Value |", "|---|---|"]
    for k, v in metrics.items():
        if k != "top_predicates":
            md.append(f"| {k} | {v} |")
    (D / f"validation_report{tag}.md").write_text("\n".join(md) + "\n",
                                                   encoding="utf-8")

    print(f"\nfindings: {sev.get('error',0):,} errors, {sev.get('warning',0):,} warnings, "
          f"{sev.get('info',0):,} info")
    for code, n in codes.most_common(10):
        print(f"  {code:28s} {n:,}")
    print(f"\nquality: richness={metrics['relationship_richness']}, "
          f"avg_degree={metrics['average_degree']}, isolated={metrics['isolated_fraction']:.1%}, "
          f"provenance={metrics['provenance_coverage']:.1%}")
    print(f"wrote {(D / f'validation_report{tag}.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
