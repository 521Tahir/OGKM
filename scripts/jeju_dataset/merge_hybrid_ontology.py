"""
Merge the offline base ontology with LLM-extracted knowledge → HYBRID ontology.

Hybrid = deterministic offline/gazetteer/geo/mined base (v1.0.0, accurate) UNION
LLM-extracted concepts + semantic relations (from llm_extract_shard.py), with
frequency thresholds to suppress noise, then the taxonomy backbone, SHACL
validation, and RDF/JSON-LD export → v1.1.0.

Usage:
  python scripts/jeju_dataset/merge_hybrid_ontology.py \
      --base 1.0.0 --target 1.1.0 --min-concept-freq 3 --min-rel-freq 2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.proctitle import set_proctitle
from src.ontology.schema import TourismOntologySchema, CLASS_HIERARCHY
from src.ontology.shacl_validator import SHACLValidator
from src.ontology.exporters import OntologyExporter
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("merge_hybrid")
ONT = ROOT / "artifacts" / "ontologies"
LLM_DIR = ROOT / "data" / "jeju_full" / "llm_onto"
ROOT_CLASS = "TourismEntity"


def _prov(conf, src):
    return {"source_record_id": -1, "extraction_prompt_version": f"hybrid/{src}",
            "timestamp": datetime.now(timezone.utc).isoformat(), "confidence": round(conf, 3),
            "validator_status": "pass", "approved_by": None, "notes": ""}


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser(description="Merge offline + LLM → hybrid ontology")
    ap.add_argument("--base", default="1.0.0")
    ap.add_argument("--target", default="1.1.0")
    ap.add_argument("--min-concept-freq", type=int, default=3)
    ap.add_argument("--min-rel-freq", type=int, default=2)
    args = ap.parse_args()

    base = ONT / args.base
    concepts = {c["name"]: c for c in load_json(base / "concepts.json")}
    relations = {(r["subject"], r["predicate"], r["object"]): r
                 for r in load_json(base / "relations.json")}
    n_base_c, n_base_r = len(concepts), len(relations)
    logger.info("Offline base v%s: %d concepts, %d relations", args.base, n_base_c, n_base_r)

    # ── aggregate LLM shard outputs ────────────────────────────────────────────
    name_lower = {n.lower(): n for n in concepts}
    c_class = Counter()        # (name, class) -> freq
    c_total = Counter()        # name -> freq
    r_freq = Counter()         # (s,p,o) -> freq
    n_recs = 0
    for sf in sorted(glob.glob(str(LLM_DIR / "llm_shard*.jsonl"))):
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_recs += 1
                rec = json.loads(line)
                for c in rec.get("concepts", []):
                    nm, cl = c.get("name", "").strip(), c.get("class", "")
                    if nm:
                        c_class[(nm, cl)] += 1
                        c_total[nm] += 1
                for r in rec.get("relations", []):
                    s, p, o = r.get("s", "").strip(), r.get("p", ""), r.get("o", "").strip()
                    if s and o:
                        r_freq[(s, p, o)] += 1
    logger.info("LLM: %d records, %d distinct concept names, %d distinct relations",
                n_recs, len(c_total), len(r_freq))

    # ── merge concepts (LLM adds new ones above threshold) ─────────────────────
    added_c = 0
    majority_class: dict[str, str] = {}
    for nm, _ in c_total.items():
        best = max(((cl, f) for (n2, cl), f in c_class.items() if n2 == nm),
                   key=lambda x: x[1], default=(None, 0))[0]
        majority_class[nm] = best
    for nm, f in c_total.items():
        if f < args.min_concept_freq:
            continue
        if nm.lower() in name_lower:
            continue  # offline/gazetteer authoritative
        cls = majority_class.get(nm) or "Attraction"
        concepts[nm] = {"name": nm, "concept_class": cls, "aliases": [], "attributes": {},
                        "provenance": _prov(min(0.85, 0.55 + f / 50), "llm")}
        name_lower[nm.lower()] = nm
        added_c += 1

    def canon(n):
        return name_lower.get(n.lower())

    # ── merge relations (LLM adds new ones above threshold, endpoints known) ───
    added_r = 0
    for (s, p, o), f in r_freq.items():
        if f < args.min_rel_freq:
            continue
        cs, co = canon(s), canon(o)
        if not cs or not co or cs == co:
            continue
        key = (cs, p, co)
        if key not in relations:
            relations[key] = {"subject": cs, "predicate": p, "object": co,
                              "provenance": _prov(min(0.85, 0.55 + f / 20), "llm")}
            added_r += 1

    # ── taxonomy backbone (is_a / subclass_of / root) ──────────────────────────
    parent_of = dict(CLASS_HIERARCHY)
    schema_classes = set(TourismOntologySchema.concept_classes)
    for cls in schema_classes:
        parent_of.setdefault(cls, ROOT_CLASS)
    for cls in schema_classes | {ROOT_CLASS}:
        if cls not in concepts:
            concepts[cls] = {"name": cls, "concept_class": "OntologyClass", "aliases": [],
                             "attributes": {}, "provenance": _prov(1.0, "backbone")}
    for cls in schema_classes:
        par = parent_of.get(cls, ROOT_CLASS)
        if cls != par:
            relations[(cls, "subclass_of", par)] = {"subject": cls, "predicate": "subclass_of",
                                                     "object": par, "provenance": _prov(1.0, "backbone")}
    for c in list(concepts.values()):
        nm, cl = c["name"], c["concept_class"]
        if cl in schema_classes and nm != cl:
            relations[(nm, "is_a", cl)] = {"subject": nm, "predicate": "is_a", "object": cl,
                                           "provenance": _prov(1.0, "backbone")}

    # ── hierarchy ──────────────────────────────────────────────────────────────
    def chain(cls):
        out, seen, cur = [], set(), cls
        while cur and cur not in seen:
            seen.add(cur); out.append(cur); cur = parent_of.get(cur)
        return list(reversed(out))
    hierarchy = [{"concept": c["name"], "concept_class": c["concept_class"],
                  "parent_class": parent_of.get(c["concept_class"], ROOT_CLASS),
                  "hierarchy_path": chain(c["concept_class"]), "is_instance": True, "confidence": 1.0}
                 for c in concepts.values() if c["concept_class"] != "OntologyClass"]

    # ── save + validate + export ───────────────────────────────────────────────
    out = ONT / args.target
    ensure_dir(out)
    concept_list, relation_list = list(concepts.values()), list(relations.values())
    save_json(concept_list, out / "concepts.json")
    save_json(relation_list, out / "relations.json")
    save_json(hierarchy, out / "hierarchy.json")
    save_json(load_json(base / "rules.json") if (base / "rules.json").exists() else [], out / "rules.json")
    schema = TourismOntologySchema.to_dict(); schema["version"] = args.target
    save_json(schema, out / "schema.json")

    rep = SHACLValidator().validate_ontology(concept_list, relation_list)
    try:
        rules_list = load_json(out / "rules.json")
        exp = OntologyExporter(output_dir=ONT)
        exp.export_turtle(concept_list, relation_list, rules_list, version=args.target,
                          output_path=ONT / f"tourism_v{args.target}.ttl")
        exp.export_jsonld(concept_list, relation_list, version=args.target,
                          output_path=ONT / f"tourism_v{args.target}.jsonld")
    except Exception as e:
        logger.warning("export skipped: %s", e)

    by_pred = Counter(r["predicate"] for r in relation_list)
    print("\n" + "=" * 64)
    print(f"HYBRID JEJU ONTOLOGY (offline + LLM)  v{args.target}")
    print("=" * 64)
    print(f"  Concepts : {len(concept_list)}  (offline {n_base_c} + LLM-added {added_c} + backbone classes)")
    print(f"  Relations: {len(relation_list)}  (offline {n_base_r} + LLM-added {added_r} + backbone)")
    print(f"  by predicate: {dict(by_pred.most_common())}")
    print(f"  Hierarchy: {len(hierarchy)} | SHACL conforms={rep.conforms}, "
          f"violations={len(rep.violations)}, warnings={len(rep.warnings)}")
    print(f"  Saved → {out.relative_to(ROOT)}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
