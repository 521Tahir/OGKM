"""
Build a proper, accurate Jeju tourism ontology.

Combines three accurate, deterministic sources (no slow/error-prone LLM extraction):
  1. Curated gazetteer  — real Jeju places with categories + lat/long, cuisine,
     activities, traveler types, seasons (ground truth).
  2. Geographic relations — coordinate-based `near` (haversine) and `located_in`
     (nearest region), so spatial links are factually correct.
  3. Record mining       — co-occurrence over the 46K Jeju dataset to add
     has_activity / served_at / suitable_for / connected_to relations, plus
     extra entities the offline pass already found.
Finally adds the taxonomy backbone (is_a / subclass_of / root) for connectivity.

Output: artifacts/ontologies/1.0.0/{concepts,relations,hierarchy,rules,schema}.json
        + Turtle / JSON-LD exports, then SHACL-validated.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ontology.schema import TourismOntologySchema, CLASS_HIERARCHY
from src.ontology.shacl_validator import SHACLValidator
from src.ontology.exporters import OntologyExporter
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("build_jeju_onto")

GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
RECORDS = ROOT / "data" / "processed" / "cleaned_records.jsonl"
OFFLINE = ROOT / "artifacts" / "ontologies" / "0.1.0"
OUT = ROOT / "artifacts" / "ontologies" / "1.0.0"
VERSION = "1.0.0"
ROOT_CLASS = "TourismEntity"
NEAR_KM = 7.0          # coordinate threshold for `near`
MIN_COOCCUR = 3        # min record co-occurrences to keep a mined relation


def _prov(rid, conf, src="gazetteer"):
    return {
        "source_record_id": rid, "extraction_prompt_version": f"jeju-build/{src}",
        "timestamp": datetime.now(timezone.utc).isoformat(), "confidence": round(conf, 3),
        "validator_status": "pass", "approved_by": None, "notes": "",
    }


def haversine(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1); dlam = math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _concept(name, cls, conf, aliases=None, attrs=None, rid=-1, src="gazetteer"):
    return {
        "name": name, "concept_class": cls, "aliases": aliases or [],
        "attributes": attrs or {}, "provenance": _prov(rid, conf, src),
    }


def _rel(s, p, o, conf, rid=-1, src="gazetteer"):
    return {"subject": s, "predicate": p, "object": o, "provenance": _prov(rid, conf, src)}


def word_present(term: str, text: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])", text) is not None


def main() -> None:
    from src.utils.proctitle import set_proctitle
    set_proctitle()
    gaz = load_json(GAZ)
    locs = gaz["locations"]
    logger.info("Gazetteer: %d places, %d cuisine, %d activities",
                len(locs), len(gaz["cuisine"]), len(gaz["activities"]))

    concepts: dict[str, dict] = {}     # name -> concept
    def add_concept(name, cls, conf, aliases=None, attrs=None, src="gazetteer", rid=-1):
        if not name or len(name) < 2:
            return
        if name not in concepts or conf > concepts[name]["provenance"]["confidence"]:
            concepts[name] = _concept(name, cls, conf, aliases, attrs, rid, src)

    # ── 1. concepts from gazetteer (accurate classes + coords) ────────────────
    # map gazetteer categories that aren't schema concept classes
    cat_map = {"Transportation": "TransportMode"}
    coords: dict[str, tuple] = {}
    regions = []  # (name, lat, lon)
    for l in locs:
        cls = cat_map.get(l["category"], l["category"])
        add_concept(l["name"], cls, 1.0, aliases=[l["note"]] if l.get("note") else [],
                    attrs={"lat": l["lat"], "lon": l["lon"]})
        coords[l["name"]] = (l["lat"], l["lon"])
        if l["category"] in ("City", "Region"):
            regions.append((l["name"], l["lat"], l["lon"]))
    for food in gaz["cuisine"]:
        add_concept(food, "Cuisine", 0.95)
    for act in gaz["activities"]:
        add_concept(act, "Activity", 0.9)
    for tt in gaz["traveler_types"]:
        add_concept(tt, "TravelerType", 0.9)
    for s in gaz["seasons"]:
        add_concept(s, "Season", 0.95)
    for d in gaz["dietary"]:
        add_concept(d, "DietaryRequirement", 0.9)
    add_concept("Jeju", "Destination", 1.0, aliases=["Jeju Island", "Jeju-do"])

    # ── 2. merge extra entities the offline pass mined from the data ──────────
    if (OFFLINE / "concepts.json").exists():
        for c in load_json(OFFLINE / "concepts.json"):
            nm = c.get("name", "")
            if nm and nm not in concepts and nm.lower() != "jeju" and len(nm) >= 3:
                add_concept(nm, c.get("concept_class", "Attraction"),
                            min(c.get("provenance", {}).get("confidence", 0.6), 0.7),
                            aliases=c.get("aliases", []), src="offline")
    logger.info("Concepts: %d", len(concepts))

    relations: dict[tuple, dict] = {}
    def add_rel(s, p, o, conf, src="gazetteer", rid=-1):
        if s and o and s != o and (s, p, o) not in relations:
            relations[(s, p, o)] = _rel(s, p, o, conf, rid, src)

    # ── 3. geographic relations from coordinates (factually correct) ──────────
    names = list(coords.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = haversine(coords[names[i]], coords[names[j]])
            if d <= NEAR_KM:
                conf = round(max(0.6, 1.0 - d / NEAR_KM * 0.4), 3)
                add_rel(names[i], "near", names[j], conf, "geo")
                add_rel(names[j], "near", names[i], conf, "geo")
    for nm, (la, lo) in coords.items():
        add_rel(nm, "located_in", "Jeju", 0.95, "geo")
        # nearest region anchor
        best = min(((rn, haversine((la, lo), (rla, rlo))) for rn, rla, rlo in regions
                    if rn != nm), key=lambda x: x[1], default=None)
        if best and best[1] <= 20:
            add_rel(nm, "part_of", best[0], round(max(0.6, 1.0 - best[1] / 25), 3), "geo")
    for rn, _, _ in regions:
        add_rel(rn, "part_of", "Jeju", 0.95, "geo")

    # ── 4. co-occurrence mining over the 46K records ──────────────────────────
    place_names = {l["name"].lower(): l["name"] for l in locs}
    act_names = {a.lower(): a for a in gaz["activities"]}
    food_names = {f.lower(): f for f in gaz["cuisine"]}
    trav_names = {t.lower(): t for t in gaz["traveler_types"]}
    co_act, co_food, co_trav, co_place = Counter(), Counter(), Counter(), Counter()
    n_rec = 0
    with open(RECORDS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rec += 1
            r = json.loads(line)
            text = f"{r.get('Input','')} {r.get('Output','')}".lower()
            pl = [orig for low, orig in place_names.items() if word_present(low, text)]
            ac = [orig for low, orig in act_names.items() if word_present(low, text)]
            fo = [orig for low, orig in food_names.items() if word_present(low, text)]
            tv = [orig for low, orig in trav_names.items() if word_present(low, text)]
            for p in pl:
                for a in ac:
                    co_act[(p, a)] += 1
                for fd in fo:
                    co_food[(fd, p)] += 1
                for t in tv:
                    co_trav[(p, t)] += 1
            for x in range(len(pl)):
                for y in range(x + 1, len(pl)):
                    co_place[tuple(sorted((pl[x], pl[y])))] += 1
    logger.info("Mined %d records: act=%d food=%d trav=%d placepairs=%d",
                n_rec, len(co_act), len(co_food), len(co_trav), len(co_place))

    def conf_of(c):
        return round(min(0.9, 0.55 + math.log1p(c) / 12), 3)
    for (p, a), c in co_act.items():
        if c >= MIN_COOCCUR:
            add_rel(p, "has_activity", a, conf_of(c), "mined")
    for (fd, p), c in co_food.items():
        if c >= MIN_COOCCUR:
            add_rel(fd, "served_at", p, conf_of(c), "mined")
    for (p, t), c in co_trav.items():
        if c >= MIN_COOCCUR:
            add_rel(p, "suitable_for", t, conf_of(c), "mined")
    for (a, b), c in co_place.items():
        if c >= MIN_COOCCUR:
            add_rel(a, "connected_to", b, conf_of(c), "mined")

    # ── 5. taxonomy backbone (is_a + subclass_of + root) ──────────────────────
    parent_of = dict(CLASS_HIERARCHY)
    schema_classes = set(TourismOntologySchema.concept_classes)
    for cls in schema_classes:
        parent_of.setdefault(cls, ROOT_CLASS)
    # class nodes
    for cls in schema_classes | {ROOT_CLASS}:
        if cls not in concepts:
            add_concept(cls, "OntologyClass", 1.0)
    for cls in schema_classes:
        par = parent_of.get(cls, ROOT_CLASS)
        if cls != par:
            add_rel(cls, "subclass_of", par, 1.0, "backbone")
    for c in list(concepts.values()):
        nm, cls = c["name"], c["concept_class"]
        if cls in schema_classes and nm != cls:
            add_rel(nm, "is_a", cls, 1.0, "backbone")

    # ── 6. hierarchy.json ──────────────────────────────────────────────────────
    def chain(cls):
        out, seen, cur = [], set(), cls
        while cur and cur not in seen:
            seen.add(cur); out.append(cur); cur = parent_of.get(cur)
        return list(reversed(out))
    hierarchy = []
    for c in concepts.values():
        cls = c["concept_class"]
        if cls == "OntologyClass":
            continue
        hierarchy.append({"concept": c["name"], "concept_class": cls,
                          "parent_class": parent_of.get(cls, ROOT_CLASS),
                          "hierarchy_path": chain(cls), "is_instance": True, "confidence": 1.0})

    # ── 7. rules (seed constraints) ───────────────────────────────────────────
    rules = load_json(OFFLINE / "rules.json") if (OFFLINE / "rules.json").exists() else []

    # ── save + validate + export ──────────────────────────────────────────────
    ensure_dir(OUT)
    concept_list = list(concepts.values())
    relation_list = list(relations.values())
    save_json(concept_list, OUT / "concepts.json")
    save_json(relation_list, OUT / "relations.json")
    save_json(hierarchy, OUT / "hierarchy.json")
    save_json(rules, OUT / "rules.json")
    schema = TourismOntologySchema.to_dict()
    schema["version"] = VERSION
    save_json(schema, OUT / "schema.json")

    rep = SHACLValidator().validate_ontology(concept_list, relation_list)

    # exports
    try:
        exp = OntologyExporter(output_dir=OUT.parent)
        exp.export_turtle(concept_list, relation_list, rules, version=VERSION,
                          output_path=OUT.parent / f"tourism_v{VERSION}.ttl")
        exp.export_jsonld(concept_list, relation_list, version=VERSION,
                          output_path=OUT.parent / f"tourism_v{VERSION}.jsonld")
    except Exception as e:
        logger.warning("export skipped: %s", e)

    by_pred = Counter(r["predicate"] for r in relation_list)
    print("\n" + "=" * 64)
    print(f"PROPER JEJU ONTOLOGY BUILT  v{VERSION}")
    print("=" * 64)
    print(f"  Concepts : {len(concept_list)}")
    print(f"  Relations: {len(relation_list)}")
    print(f"  by predicate: {dict(by_pred.most_common())}")
    print(f"  Hierarchy: {len(hierarchy)} nodes | Rules: {len(rules)}")
    print(f"  SHACL    : conforms={rep.conforms}, violations={len(rep.violations)}, warnings={len(rep.warnings)}")
    print(f"  Saved → {OUT.relative_to(ROOT)}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
