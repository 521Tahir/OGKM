"""
Stage 2 — the 200-item Jeju benchmark: 100 multi-hop + 100 single-hop.

Grounded in `data/jeju_full/jeju_gazetteer.json` — the one curated, non-synthetic
artifact in the project (69 real places with coordinates, categories and notes).

Why not the ontology or the corpus:
  * 59.9% of type-checkable ontology relations are implausible
    ("Teddy Bear Museum has_activity hiking", 750 bad `served_at` triples), so
    gold answers drawn from it would score models on reproducing errors;
  * the 46k corpus is itself LLM-generated, so corpus co-occurrence attests
    nothing — and gold drawn from it would favour the RAG arm by construction.
The gazetteer is neutral to both retrievers and actually correct.

Multi-hop items chain two gazetteer facts and never name the bridge entity, so
the question cannot be answered by a single lookup. Every answer is computed
(haversine over real coordinates), so the gold is auditable rather than asserted.

Emits outputs/jeju_bench_200.jsonl with the same {record_id, Input, Output}
schema the exp024 harness already consumes, plus provenance fields.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.jeju_guard import jeju_guard
from src.utils.proctitle import set_proctitle

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
SEED = 20260911
N_MULTI, N_SINGLE = 100, 100

ARTICLE = {"Mountain": "mountain", "NaturalAttraction": "natural attraction",
           "Beach": "beach", "Island": "island", "Museum": "museum",
           "CulturalAttraction": "cultural attraction", "ReligiousSite": "temple",
           "Attraction": "attraction", "Trail": "trail", "City": "city",
           "Region": "town or region", "Market": "traditional market",
           "Transportation": "transport hub"}


def haversine(a: dict, b: dict) -> float:
    R = 6371.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp = p2 - p1
    dl = math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def main() -> None:
    set_proctitle()
    rng = random.Random(SEED)
    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    locs = [x for x in gaz["locations"] if isinstance(x, dict) and "lat" in x]
    by_name = {x["name"]: x for x in locs}
    by_cat: dict[str, list[dict]] = {}
    for x in locs:
        by_cat.setdefault(x["category"], []).append(x)

    def nearest(anchor: dict, category=None, exclude: set = None) -> dict | None:
        exclude = (exclude or set()) | {anchor["name"]}
        cats = [category] if isinstance(category, str) else category
        pool = [y for c in cats for y in by_cat.get(c, [])] if cats else locs
        cand = [x for x in pool if x["name"] not in exclude]
        return min(cand, key=lambda x: haversine(anchor, x)) if cand else None

    def describe(x: dict) -> str:
        """Natural apposition for a gazetteer note. Notes are a mix of features
        ("UNESCO lava tube") and bare locators ("Jeju City"), so "known for its
        {note}" produces "known for its Jeju City". Use a neutral dash instead."""
        return f"{x['name']} ({x['note']})"

    items: list[dict] = []

    def add(kind: str, template: str, question: str, answer: str, chain: list[str]) -> None:
        ok, _ = jeju_guard(question)
        if not ok:
            return
        items.append({"hops": 2 if kind == "multihop" else 1, "kind": kind,
                      "template": template, "Input": question.strip(),
                      "Output": answer.strip(), "evidence": chain})

    # ══ MULTI-HOP ═════════════════════════════════════════════════════════
    # T1 proximity -> attribute. Bridge (the nearest X) is never named.
    for anchor in locs:
        for cat in ("Museum", "Beach", "Trail", "Market", "Island"):
            if anchor["category"] == cat:
                continue
            b = nearest(anchor, cat)
            if not b or haversine(anchor, b) > 30:
                continue
            add("multihop", "T1_proximity_attribute",
                f"Which {ARTICLE[cat]} on Jeju Island lies closest to {anchor['name']}, "
                f"and what is that place known for?",
                f"The closest {ARTICLE[cat]} to {anchor['name']} is {b['name']}, about "
                f"{haversine(anchor, b):.0f} km away — {b['note']}.",
                [f"{anchor['name']} @ ({anchor['lat']}, {anchor['lon']})",
                 f"nearest {cat} = {b['name']} ({haversine(anchor, b):.1f} km)",
                 f"{b['name']} note = {b['note']}"])

    # T2 anchor -> its town/region -> another attraction in that region
    for anchor in locs:
        if anchor["category"] in ("Region", "City"):
            continue
        reg = nearest(anchor, ["Region", "City"])
        if not reg:
            continue
        c = nearest(reg, "NaturalAttraction", exclude={anchor["name"]})
        if not c or haversine(reg, c) > 25:
            continue
        add("multihop", "T2_region_attraction",
            f"{anchor['name']} sits closest to which town on Jeju, and which natural "
            f"attraction would I find near that same town?",
            f"{anchor['name']} is closest to {reg['name']} — {reg['note']}. Near "
            f"{reg['name']} you can visit {c['name']} — {c['note']}.",
            [f"nearest region to {anchor['name']} = {reg['name']}",
             f"nearest NaturalAttraction to {reg['name']} = {c['name']}",
             f"{c['name']} note = {c['note']}"])

    # T3 superlative/descriptor -> related place (descriptor resolves the bridge)
    DESCRIPTORS = [
        ("the highest peak in South Korea", "Hallasan"),
        ("the southernmost point of Korea", "Marado Island"),
        ("Jeju's UNESCO-listed lava tube", "Manjanggul Cave"),
        ("the UNESCO tuff cone known as Sunrise Peak", "Seongsan Ilchulbong"),
        ("Jeju's columnar basalt cliffs", "Jusangjeolli Cliffs"),
        ("the lava dome with a grotto temple", "Sanbangsan"),
        ("Jeju's ancient nutmeg forest", "Bijarim Forest"),
        ("the volcanic crater famous for silver grass", "Sangumburi Crater"),
        ("Jeju's green tea museum", "O'sulloc Tea Museum"),
        ("the museum devoted to Jeju's women divers", "Jeju Haenyeo Museum"),
        ("Jeju's main airport", "Jeju International Airport"),
        ("the preserved historic village on Jeju", "Seongeup Folk Village"),
    ]
    for desc, name in DESCRIPTORS:
        a = by_name.get(name)
        if not a:
            continue
        for cat in ("Beach", "Museum", "Trail", "Market", "City", "Region"):
            if a["category"] == cat:
                continue
            b = nearest(a, cat)
            if not b or haversine(a, b) > 35:
                continue
            add("multihop", "T3_descriptor_proximity",
                f"Which {ARTICLE[cat]} on Jeju is nearest to {desc}, and what is it noted for?",
                f"{desc[0].upper() + desc[1:]} is {a['name']}. The nearest {ARTICLE[cat]} is "
                f"{b['name']}, roughly {haversine(a, b):.0f} km away — {b['note']}.",
                [f"{desc} = {a['name']}", f"nearest {cat} = {b['name']}",
                 f"{b['name']} note = {b['note']}"])

    # T4 two-step proximity: anchor -> nearest trail -> that trail's nearest city
    for anchor in locs:
        if anchor["category"] == "Trail":
            continue
        t = nearest(anchor, "Trail")
        if not t or haversine(anchor, t) > 25:
            continue
        city = nearest(t, "City")
        if not city:
            continue
        add("multihop", "T4_trail_city",
            f"If I hike the trail closest to {anchor['name']}, which of Jeju's two "
            f"cities would I be nearest to afterwards?",
            f"The trail closest to {anchor['name']} is {t['name']} ({t['note']}), and "
            f"the nearer city is {city['name']} — {city['note']}.",
            [f"nearest Trail to {anchor['name']} = {t['name']}",
             f"nearest City to {t['name']} = {city['name']}"])

    multihop = [x for x in items if x["kind"] == "multihop"]

    # ══ SINGLE-HOP ════════════════════════════════════════════════════════
    items = []
    for x in locs:
        add("single", "S1_what_is",
            f"What kind of place is {x['name']} on Jeju Island, and what is it known for?",
            f"{x['name']} is a {ARTICLE[x['category']]} on Jeju Island — {x['note']}.",
            [f"{x['name']} category = {x['category']}", f"note = {x['note']}"])
        c = nearest(x, "City")
        if c:
            add("single", "S2_which_city",
                f"Which of Jeju's cities is {x['name']} closest to?",
                f"{x['name']} is closest to {c['name']} — {c['note']}.",
                [f"nearest City to {x['name']} = {c['name']} ({haversine(x, c):.1f} km)"])
        r = nearest(x, "Region")
        if r and x["category"] != "Region":
            add("single", "S3_which_region",
                f"Setting aside Jeju City and Seogwipo, which of Jeju's towns or "
                f"regions is {x['name']} nearest to?",
                f"{x['name']} is nearest to {r['name']} — {r['note']}.",
                [f"nearest Region to {x['name']} = {r['name']} ({haversine(x, r):.1f} km)"])
    single = [x for x in items if x["kind"] == "single"]

    # ══ sample, balancing templates ═══════════════════════════════════════
    def take(pool: list[dict], n: int) -> list[dict]:
        buckets: dict[str, list[dict]] = {}
        for x in pool:
            buckets.setdefault(x["template"], []).append(x)
        for v in buckets.values():
            rng.shuffle(v)
        out, keys = [], sorted(buckets)
        while len(out) < n and any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k] and len(out) < n:
                    out.append(buckets[k].pop())
        return out

    rng.shuffle(multihop); rng.shuffle(single)
    chosen = take(multihop, N_MULTI) + take(single, N_SINGLE)
    for i, x in enumerate(chosen):
        x["record_id"] = 900000 + i

    path = OUT / "jeju_bench_200.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for x in chosen:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    import collections
    print(f"wrote {len(chosen)} items -> {path}")
    print("  by kind:", collections.Counter(x["kind"] for x in chosen))
    print("  by template:", dict(collections.Counter(x["template"] for x in chosen)))
    print(f"  pools available: multihop={len(multihop)}, single={len(single)}")


if __name__ == "__main__":
    main()
