"""
Goal 3, lever 1 — remove factually implausible relations from the ontology.

Diagnosis (exp025): 59.9% of type-checkable relations are wrong, e.g.
  "Teddy Bear Museum has_activity hiking"
  "Jeju black pork served_at Baengnokdam"   (a summit crater lake)
  "Yeongsil Trail located_in Namwon"        (the trail is on Hallasan)
The ontology arm has therefore been injecting false statements into every prompt.

This builds ontology v1.3.0 by type-checking every relation against the entity's
category and dropping the violations. The gazetteer is used ONLY as a type
vocabulary (a validator), never as a source of new facts — adding its facts would
leak the exp025 gold, which is derived from it.

  python -m experiments.exp026_onto_improve.clean_ontology
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.proctitle import set_proctitle

SRC_VER, DST_VER = "1.2.0", "1.3.0"
ONTO = ROOT / "artifacts" / "ontologies"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── type vocabulary ───────────────────────────────────────────────────────
COASTAL = {"Beach", "Island", "Coast"}
UPLAND = {"Mountain", "Trail", "Oreum", "Forest"}
SETTLEMENT = {"City", "Region", "Market", "Restaurant", "Accommodation", "Village"}
INDOOR = {"Museum", "CulturalAttraction", "ReligiousSite", "Attraction", "Landmark"}

WATER_ACT = {"snorkeling", "scuba diving", "surfing", "sea kayaking", "submarine tour",
             "haenyeo (women diver) show", "swimming", "diving", "kayaking",
             "jet ski", "parasailing", "boat tour", "fishing"}
UPLAND_ACT = {"hiking", "olle trail walking", "horseback riding", "stargazing",
              "trekking", "climbing", "camping", "cycling", "hiking trail",
              "sunrise viewing", "nature walk", "forest bathing"}
INDOOR_ACT = {"tea ceremony", "cafe hopping", "museum visit", "shopping",
              "art exhibition", "pottery"}

# predicate -> (allowed subject kinds, allowed object kinds); None = unconstrained
PLACE_LIKE = COASTAL | UPLAND | SETTLEMENT | INDOOR | {"NaturalAttraction",
                                                       "Destination", "Attraction"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main() -> None:
    set_proctitle()
    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    cat: dict[str, str] = {}
    for x in gaz["locations"]:
        if isinstance(x, dict):
            cat[norm(x["name"])] = x["category"]
    acts = {norm(a if isinstance(a, str) else a["name"]) for a in gaz["activities"]}
    cuis = {norm(c if isinstance(c, str) else c["name"]) for c in gaz["cuisine"]}

    src = ONTO / SRC_VER
    rels = json.loads((src / "relations.json").read_text(encoding="utf-8"))
    cons = json.loads((src / "concepts.json").read_text(encoding="utf-8"))
    ocls = {norm(c["name"]): c.get("concept_class") for c in cons if c.get("name")}

    def kind(name: str) -> str | None:
        """Best known category for an entity: gazetteer first, then ontology class."""
        n = norm(name)
        return cat.get(n) or ocls.get(n)

    def is_activity(name: str) -> bool:
        n = norm(name)
        return n in acts or ocls.get(n) == "Activity"

    def is_food(name: str) -> bool:
        n = norm(name)
        return n in cuis or ocls.get(n) == "Cuisine"

    reasons = Counter()

    def violates(s: str, p: str, o: str) -> str | None:
        ks, ko = kind(s), kind(o)

        if p == "has_activity" and is_activity(o):
            a = norm(o)
            if ks in COASTAL and a in UPLAND_ACT - {"cycling", "sunrise viewing",
                                                    "nature walk", "horseback riding"}:
                return "water_site_upland_activity"
            if ks in UPLAND and a in WATER_ACT:
                return "upland_site_water_activity"
            if ks in INDOOR and (a in WATER_ACT or a in UPLAND_ACT):
                return "indoor_site_outdoor_activity"
            if ks == "Market" and (a in WATER_ACT or a in UPLAND_ACT):
                return "market_outdoor_activity"

        if p == "served_at" and is_food(s):
            # food is served in settlements, not on a summit or a trail
            if ko in UPLAND or ko == "NaturalAttraction":
                return "food_served_at_wilderness"
            if ko in INDOOR - {"Attraction"}:
                return "food_served_at_museum_or_temple"

        if p == "located_in":
            # a place is located in a settlement/region, not in an activity or dish
            if is_activity(o) or is_food(o):
                return "located_in_non_place"
            if ks in PLACE_LIKE and ko and ko not in (SETTLEMENT | {"Island", "Region",
                                                                   "City", "Destination",
                                                                   "Mountain"}):
                return "located_in_non_container"

        if p == "reachable_by" and (is_food(o) or is_activity(o)):
            return "reachable_by_non_transport"

        if p in ("near", "connected_to") and (is_food(s) or is_food(o)
                                              or is_activity(s) or is_activity(o)):
            return "spatial_relation_on_non_place"

        return None

    kept, dropped = [], []
    for r in rels:
        s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
        if not s or not o or norm(s) == norm(o):
            reasons["degenerate"] += 1
            dropped.append({**r, "_reason": "degenerate"})
            continue
        why = violates(s, p, o)
        if why:
            reasons[why] += 1
            dropped.append({**r, "_reason": why})
        else:
            kept.append(r)

    dst = ONTO / DST_VER
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("concepts.json", "hierarchy.json", "rules.json", "schema.json",
               "temporal_facts.json"):
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)
    (dst / "relations.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    (OUT / "dropped_relations.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dropped), encoding="utf-8")

    n0 = len(rels)
    print(f"ontology v{SRC_VER} -> v{DST_VER}")
    print(f"  relations {n0:,} -> {len(kept):,}  (dropped {len(dropped):,} = "
          f"{100*len(dropped)/n0:.1f}%)")
    for why, c in reasons.most_common():
        print(f"    {why:36s} {c:6,}")
    print("\n  examples of dropped relations:")
    for d in dropped[:8]:
        print(f"    [{d['_reason']}] {d.get('subject')} -{d.get('predicate')}-> {d.get('object')}")

    (OUT / "clean_report.json").write_text(json.dumps({
        "src_version": SRC_VER, "dst_version": DST_VER,
        "relations_before": n0, "relations_after": len(kept),
        "dropped": len(dropped), "drop_pct": round(100 * len(dropped) / n0, 2),
        "reasons": dict(reasons)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
