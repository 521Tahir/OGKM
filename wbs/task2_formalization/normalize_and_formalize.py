"""
WBS Task 2.2.1 - Ontology normalization, formalization and RDF/OWL conversion.

Operates on the real Jeju ontology (artifacts/ontologies/<version>/) and produces:
  * a normalized ontology    (synonyms merged, duplicates collapsed, canonical names)
  * Turtle + RDF/XML + JSON-LD serialisations with a real OWL TBox
  * a normalization report recording every merge decision, so the mapping is auditable

Normalization rules, in order of application. Each is conservative: a merge happens
only when the evidence is unambiguous, because silently collapsing two genuinely
different Jeju places would corrupt the ontology rather than clean it.

  1. whitespace / punctuation / case folding for the match key only
  2. alias resolution      - a concept whose name appears in another concept's
                             `aliases` list is merged into that concept
  3. canonical casing      - the surface form with the highest in-corpus frequency
                             wins; ties break toward the gazetteer spelling
  4. duplicate collapse    - identical match keys merge, union-ing their relations,
                             aliases and attributes; provenance is concatenated
  5. near-duplicate review - keys differing only by a trailing descriptor
                             ("X", "X Cave") are REPORTED, never auto-merged

  python -m wbs.task2_formalization.normalize_and_formalize --version 1.3.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NS = "http://jejunu.ac.kr/ontology/jeju-tourism#"
PLACE_SUFFIX = re.compile(
    r"\b(cave|beach|museum|market|temple|trail|oreum|peak|island|village|garden|"
    r"forest|rock|cliff|coast|harbor|harbour|port|airport|tower|park|falls|"
    r"waterfall|mountain|resort|arboretum|observatory|hill|valley|bridge|lake)$",
    re.I)


def key(s: str) -> str:
    """Match key: case/space/punctuation-insensitive, used only for comparison."""
    s = re.sub(r"[’'`]", "'", str(s or ""))
    s = re.sub(r"[^\w\s'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load(version: str) -> tuple[list, list, list]:
    d = ROOT / "artifacts" / "ontologies" / version
    rd = lambda f: json.loads((d / f).read_text(encoding="utf-8")) if (d / f).exists() else []
    return rd("concepts.json"), rd("relations.json"), rd("hierarchy.json")


def corpus_frequency() -> Counter:
    """Surface-form frequency over the TRAIN split, used to pick canonical casing."""
    freq = Counter()
    p = ROOT / "data" / "jeju_full" / "splits" / "train.jsonl"
    if not p.exists():
        return freq
    pat = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:[ \-][A-Z][a-zA-Z'\-]+){0,3}\b")
    for line in open(p, encoding="utf-8"):
        try:
            out = json.loads(line)["Output"]
        except Exception:
            continue
        for m in pat.findall(out):
            freq[m.strip()] += 1
    return freq


def normalize(concepts, relations, hierarchy, freq: Counter):
    report = {"input": {"concepts": len(concepts), "relations": len(relations)},
              "alias_merges": [], "duplicate_merges": [], "canonical_recasing": [],
              "near_duplicates_flagged": [], "dropped_malformed": []}

    # ── index concepts by match key ────────────────────────────────────────
    by_key: dict[str, list[dict]] = defaultdict(list)
    for c in concepts:
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            report["dropped_malformed"].append(str(c)[:120])
            continue
        by_key[key(c["name"])].append(c)

    # ── rule 2: alias resolution ───────────────────────────────────────────
    alias_owner: dict[str, str] = {}
    for c in concepts:
        if not isinstance(c, dict):
            continue
        for a in (c.get("aliases") or []):
            ak = key(a)
            if ak and ak != key(c["name"]) and ak not in alias_owner:
                alias_owner[ak] = key(c["name"])

    merged: dict[str, dict] = {}
    remap: dict[str, str] = {}          # old match key -> surviving match key

    for k, group in by_key.items():
        target = alias_owner.get(k)
        if target and target in by_key and target != k:
            remap[k] = target
            report["alias_merges"].append({"merged": group[0]["name"],
                                           "into": by_key[target][0]["name"],
                                           "reason": "name appears in target's aliases"})
            continue
        # rule 3: canonical casing by corpus frequency, gazetteer spelling breaks ties
        forms = [g["name"] for g in group]
        canon = max(forms, key=lambda f: (freq.get(f, 0), f == forms[0], -len(f)))
        if canon != forms[0]:
            report["canonical_recasing"].append({"from": forms[0], "to": canon,
                                                 "corpus_freq": freq.get(canon, 0)})
        # rule 4: duplicate collapse - union everything
        base = dict(group[0]); base["name"] = canon
        if len(group) > 1:
            al, attrs, provs = set(base.get("aliases") or []), dict(base.get("attributes") or {}), []
            for g in group:
                al.update(g.get("aliases") or [])
                attrs.update(g.get("attributes") or {})
                if g.get("provenance"):
                    provs.append(g["provenance"])
            al.discard(canon)
            base["aliases"] = sorted(al)
            base["attributes"] = attrs
            base["provenance"] = provs[0] if provs else base.get("provenance")
            base["merged_from"] = len(group)
            report["duplicate_merges"].append({"canonical": canon, "n_merged": len(group),
                                               "surfaces": forms})
        merged[k] = base
        remap[k] = k

    # resolve alias chains
    for k in list(remap):
        seen = set()
        t = remap[k]
        while t in remap and remap[t] != t and t not in seen:
            seen.add(t); t = remap[t]
        remap[k] = t

    # ── rule 5: flag near-duplicates, never auto-merge ─────────────────────
    keys = sorted(merged)
    for k in keys:
        m = PLACE_SUFFIX.search(k)
        if not m:
            continue
        stem = k[:m.start()].strip()
        if stem and stem in merged:
            report["near_duplicates_flagged"].append(
                {"specific": merged[k]["name"], "generic": merged[stem]["name"],
                 "action": "NOT merged - requires expert review"})

    # ── rewrite relations onto surviving concepts ──────────────────────────
    name_of = {k: v["name"] for k, v in merged.items()}
    out_rel, seen_rel, dangling = [], set(), 0
    for r in relations:
        if not isinstance(r, dict):
            continue
        s, p, o = r.get("subject"), r.get("predicate"), r.get("object")
        if not (s and p and o):
            continue
        sk, ok = remap.get(key(s), key(s)), remap.get(key(o), key(o))
        s2, o2 = name_of.get(sk, s), name_of.get(ok, o)
        if key(s2) == key(o2):
            continue                                   # self-loop after merge
        sig = (key(s2), p, key(o2))
        if sig in seen_rel:
            continue
        seen_rel.add(sig)
        if sk not in merged and ok not in merged:
            dangling += 1
        out_rel.append({**r, "subject": s2, "object": o2})

    # ── rewrite hierarchy ──────────────────────────────────────────────────
    out_hier, seen_h = [], set()
    for h in hierarchy:
        if not isinstance(h, dict) or not h.get("concept"):
            continue
        ck = remap.get(key(h["concept"]), key(h["concept"]))
        nm = name_of.get(ck)
        if not nm:
            continue
        sig = (key(nm), h.get("concept_class"))
        if sig in seen_h:
            continue
        seen_h.add(sig)
        out_hier.append({**h, "concept": nm})

    report["output"] = {"concepts": len(merged), "relations": len(out_rel),
                        "hierarchy": len(out_hier),
                        "relations_deduplicated": len(relations) - len(out_rel),
                        "dangling_endpoints": dangling}
    return list(merged.values()), out_rel, out_hier, report


def to_rdf(concepts, relations, hierarchy, version: str):
    """Build a real OWL ontology: TBox from the class tree, ABox from the concepts."""
    from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef
    from rdflib.namespace import DCTERMS, SKOS, PROV

    J = Namespace(NS)
    g = Graph()
    g.bind("jeju", J); g.bind("owl", OWL); g.bind("skos", SKOS)
    g.bind("prov", PROV); g.bind("dcterms", DCTERMS)

    onto = URIRef(NS.rstrip("#"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, DCTERMS.title, Literal("Jeju Tourism Ontology")))
    g.add((onto, OWL.versionInfo, Literal(version)))
    g.add((onto, DCTERMS.creator, Literal("NLP-Lab, Jeju National University")))

    def uri(name: str) -> URIRef:
        slug = re.sub(r"[^\w\-]", "_", re.sub(r"\s+", "_", str(name).strip()))
        return URIRef(NS + slug)

    # TBox: class tree from hierarchy_path
    classes, class_edges = set(), set()
    for h in hierarchy:
        path = h.get("hierarchy_path") or []
        for a, b in zip(path, path[1:]):
            classes.add(a); classes.add(b); class_edges.add((a, b))
        if h.get("concept_class"):
            classes.add(h["concept_class"])
    for c in classes:
        g.add((uri(c), RDF.type, OWL.Class))
        g.add((uri(c), RDFS.label, Literal(c)))
    for parent, child in class_edges:
        g.add((uri(child), RDFS.subClassOf, uri(parent)))

    # object properties, declared once with domain/range left open (the data is
    # not clean enough to assert global domains without producing false entailments)
    preds = sorted({r["predicate"] for r in relations if r.get("predicate")})
    for p in preds:
        pu = uri(p)
        g.add((pu, RDF.type, OWL.ObjectProperty))
        g.add((pu, RDFS.label, Literal(p.replace("_", " "))))
    for sym in ("near", "co_occurs_with", "connected_to"):
        if sym in preds:
            g.add((uri(sym), RDF.type, OWL.SymmetricProperty))

    # ABox
    cls_of = {key(h["concept"]): h.get("concept_class") for h in hierarchy if h.get("concept")}
    for c in concepts:
        cu = uri(c["name"])
        klass = c.get("concept_class") or cls_of.get(key(c["name"]))
        g.add((cu, RDF.type, OWL.NamedIndividual))
        if klass:
            g.add((cu, RDF.type, uri(klass)))
        g.add((cu, RDFS.label, Literal(c["name"])))
        g.add((cu, SKOS.prefLabel, Literal(c["name"])))
        for a in (c.get("aliases") or []):
            g.add((cu, SKOS.altLabel, Literal(a)))
        for k, v in (c.get("attributes") or {}).items():
            dt = XSD.double if isinstance(v, (int, float)) else XSD.string
            g.add((cu, uri("attr_" + k), Literal(v, datatype=dt)))
        prov = c.get("provenance") or {}
        if isinstance(prov, dict) and prov.get("confidence") is not None:
            g.add((cu, uri("confidence"), Literal(float(prov["confidence"]), datatype=XSD.double)))
        if isinstance(prov, dict) and prov.get("extraction_prompt_version"):
            g.add((cu, PROV.wasGeneratedBy, Literal(str(prov["extraction_prompt_version"]))))

    for r in relations:
        g.add((uri(r["subject"]), uri(r["predicate"]), uri(r["object"])))
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.3.0")
    ap.add_argument("--out-version", default=None,
                    help="version tag for the normalized output (default <version>-norm)")
    a = ap.parse_args()
    outv = a.out_version or f"{a.version}-norm"

    concepts, relations, hierarchy = load(a.version)
    print(f"loaded v{a.version}: {len(concepts):,} concepts, {len(relations):,} relations, "
          f"{len(hierarchy):,} hierarchy records")

    freq = corpus_frequency()
    print(f"corpus surface forms for canonical casing: {len(freq):,}")

    nc, nr, nh, report = normalize(concepts, relations, hierarchy, freq)
    print(f"normalized -> {len(nc):,} concepts, {len(nr):,} relations, {len(nh):,} hierarchy")
    print(f"  alias merges        {len(report['alias_merges']):,}")
    print(f"  duplicate merges    {len(report['duplicate_merges']):,}")
    print(f"  canonical recasing  {len(report['canonical_recasing']):,}")
    print(f"  near-dups flagged   {len(report['near_duplicates_flagged']):,} (not merged)")
    print(f"  relations dedup'd   {report['output']['relations_deduplicated']:,}")

    od = ROOT / "artifacts" / "ontologies" / outv
    od.mkdir(parents=True, exist_ok=True)
    (od / "concepts.json").write_text(json.dumps(nc, indent=2, ensure_ascii=False), encoding="utf-8")
    (od / "relations.json").write_text(json.dumps(nr, indent=2, ensure_ascii=False), encoding="utf-8")
    (od / "hierarchy.json").write_text(json.dumps(nh, indent=2, ensure_ascii=False), encoding="utf-8")

    D = ROOT / "wbs" / "deliverables" / "task2_formalization"
    D.mkdir(parents=True, exist_ok=True)
    (D / "normalization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    g = to_rdf(nc, nr, nh, outv)
    print(f"\nRDF graph: {len(g):,} triples")
    for fmt, ext in (("turtle", "ttl"), ("xml", "owl"), ("json-ld", "jsonld")):
        p = D / f"jeju_tourism_ontology.{ext}"
        g.serialize(destination=str(p), format=fmt)
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")
    (od / "jeju_tourism_ontology.ttl").write_text(
        (D / "jeju_tourism_ontology.ttl").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nnormalized ontology written to artifacts/ontologies/{outv}/")


if __name__ == "__main__":
    main()
