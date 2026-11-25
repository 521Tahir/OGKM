"""
Rebuild the Jeju ontology from the FULL training split (exp031 -> v3.0.0).

Why: ontology v1.2.0 holds 1,573 concepts, but the 36,847 training answers contain
1,934 entities occurring >= 10 times, and 76-80% of those are absent from it. The
ontology arm therefore retrieves 9-19% of the gold entities on the free-form test
split while Simple RAG retrieves 73%. That gap is an artefact of under-extraction,
not a property of ontology grounding.

Fairness: this reads ONLY `splits/train.jsonl` - exactly the corpus the RAG arm
indexes. No test or val data is touched, so neither arm sees the evaluation set.

What it extracts
  is_a             entity -> class, from gazetteer categories and head-noun rules
  located_in       entity -> region/city, from surface patterns in the corpus
  has_activity     entity -> activity, from verb/noun co-occurrence
  served_at        dish   -> place
  co_occurs_with   entity -> entity, when two entities appear in the same itinerary
                   far more often than chance (PMI-filtered). This is the relation
                   the free-form task actually needs: "these places go together".
  suitable_for     entity -> traveller type / purpose

  python -m experiments.exp031_onto_rebuild.build_rich_ontology
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.proctitle import set_proctitle

TRAIN = ROOT / "data" / "jeju_full" / "splits" / "train.jsonl"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
SRC_ONTO = ROOT / "artifacts" / "ontologies" / "1.2.0"
DST = ROOT / "artifacts" / "ontologies" / "3.0.0"
OUT = Path(__file__).resolve().parent / "outputs"

MIN_ENT_FREQ = 8
MIN_PAIR_FREQ = 12
MIN_PMI = 0.8
MAX_COOC_PER_ENT = 12

# Time-of-day and itinerary scaffolding words are capitalised throughout the
# corpus but carry no factual content; they must not become ontology entities.
BOILER = {
    "morning", "afternoon", "evening", "night", "midday", "noon", "dawn", "dusk",
    "breakfast", "lunch", "dinner", "snack", "snacks", "brunch", "meal", "meals",
    "day", "days", "start", "arrival", "arrive", "departure", "depart", "return",
    "visit", "explore", "enjoy", "experience", "discover", "head", "take", "taking",
    "continue", "end", "conclude", "begin", "check", "stop", "travel", "optional",
    "afterward", "afterwards", "overnight", "transfer", "relax", "rest", "free time",
    "activity", "activities", "itinerary", "tip", "tips", "note", "notes", "option",
    "options", "recommendation", "recommendations", "highlights", "overview",
    "jeju", "jeju island", "island", "korea", "south korea", "korean", "day trip",
    "tour", "tours", "guide", "local", "locals", "traveler", "travelers", "traveller",
    "visitor", "visitors", "tourist", "tourists", "hotel", "hotels", "restaurant",
    "restaurants", "cafe", "cafes", "market", "beach", "trail", "museum", "temple",
}

HEAD_CLASS = [
    (("beach", "haebyeon"), "Beach"),
    (("waterfall", "falls", "pokpo"), "NaturalAttraction"),
    (("cave", "lava tube", "gul"), "NaturalAttraction"),
    (("oreum", "peak", "crater", "mountain", "san", "bong"), "Mountain"),
    (("trail", "olle", "course", "path"), "Trail"),
    (("museum", "gallery", "art"), "Museum"),
    (("temple", "sa", "shrine", "church"), "ReligiousSite"),
    (("market", "sijang"), "Market"),
    (("park", "garden", "arboretum", "forest", "hill"), "Attraction"),
    (("island", "do"), "Island"),
    (("village", "folk"), "CulturalAttraction"),
    (("airport", "terminal", "port", "bus", "ferry"), "TransportMode"),
    (("hotel", "resort", "guesthouse", "pension", "stay"), "Accommodation"),
    (("restaurant", "eatery", "diner", "bar", "cafe"), "Restaurant"),
    (("city", "si", "town", "eup", "myeon", "region"), "Region"),
]

ACTIVITY_WORDS = {
    "hiking", "walking", "swimming", "snorkeling", "scuba diving", "diving", "surfing",
    "kayaking", "cycling", "biking", "horseback riding", "fishing", "stargazing",
    "sunrise viewing", "sunset viewing", "photography", "shopping", "tea ceremony",
    "cafe hopping", "birdwatching", "camping", "picnicking", "sightseeing",
    "tangerine picking", "boat tour", "submarine tour", "spa", "hot spring",
}
TRAVELER_WORDS = {
    "families", "family", "couples", "solo travelers", "seniors", "elderly",
    "children", "kids", "toddlers", "backpackers", "students", "honeymooners",
    "photographers", "hikers", "foodies", "groups", "budget travelers",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def key(s: str) -> str:
    return norm(s).lower()


def classify(name: str, gaz_cat: dict) -> str | None:
    k = key(name)
    if k in gaz_cat:
        return gaz_cat[k]
    for heads, cls in HEAD_CLASS:
        for h in heads:
            if k.endswith(" " + h) or k == h or f" {h} " in f" {k} ":
                return cls
    return None


def main() -> None:
    set_proctitle()
    OUT.mkdir(parents=True, exist_ok=True)
    DST.mkdir(parents=True, exist_ok=True)

    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    gaz_cat = {key(x["name"]): x["category"] for x in gaz["locations"] if isinstance(x, dict)}
    gaz_note = {key(x["name"]): x.get("note", "") for x in gaz["locations"] if isinstance(x, dict)}

    print(f"reading {TRAIN.name} (train split only - no test/val)")
    records = []
    for line in open(TRAIN, encoding="utf-8"):
        r = json.loads(line)
        records.append(r["Output"])
    print(f"  {len(records):,} training answers")

    # ── 1. entity inventory ───────────────────────────────────────────────
    ENT_RE = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:[ \-][A-Z][a-zA-Z'\-]+){0,4}\b")
    freq = Counter()
    per_doc: list[set[str]] = []
    for text in records:
        found = set()
        for ph in ENT_RE.findall(text):
            ph = norm(ph)
            k = key(ph)
            if len(k) < 5 or k in BOILER:
                continue
            # strip a leading scaffolding word ("Visit Manjanggul Cave")
            parts = ph.split()
            while parts and key(parts[0]) in BOILER:
                parts = parts[1:]
            if not parts:
                continue
            ph = " ".join(parts)
            k = key(ph)
            if len(k) < 5 or k in BOILER:
                continue
            found.add(ph)
        for e in found:
            freq[e] += 1
        per_doc.append(found)

    ents = {e for e, c in freq.items() if c >= MIN_ENT_FREQ}
    # collapse case variants onto the most frequent surface form
    canon: dict[str, str] = {}
    by_key = defaultdict(list)
    for e in ents:
        by_key[key(e)].append(e)
    for k, variants in by_key.items():
        canon[k] = max(variants, key=lambda x: freq[x])
    ents = set(canon.values())
    print(f"  entities with freq >= {MIN_ENT_FREQ}: {len(ents):,}")

    # ── 2. typing ─────────────────────────────────────────────────────────
    concepts, classed = [], 0
    for e in sorted(ents, key=lambda x: -freq[x]):
        cls = classify(e, gaz_cat)
        if cls:
            classed += 1
        concepts.append({
            "name": e,
            "concept_class": cls or "Concept",
            "aliases": [gaz_note[key(e)]] if key(e) in gaz_note and gaz_note[key(e)] else [],
            "attributes": {"corpus_frequency": freq[e]},
            "provenance": {"source": "train_split_extraction", "version": "3.0.0"},
        })
    print(f"  typed by gazetteer/head-noun: {classed:,} / {len(concepts):,}")

    # ── 3. relations ──────────────────────────────────────────────────────
    rels = []
    seen = set()

    def add(s, p, o, conf=0.7):
        k = (key(s), p, key(o))
        if k in seen or key(s) == key(o):
            return
        seen.add(k)
        rels.append({"subject": s, "predicate": p, "object": o,
                     "provenance": {"confidence": round(conf, 3), "source": "train_split"}})

    for c in concepts:
        if c["concept_class"] != "Concept":
            add(c["name"], "is_a", c["concept_class"], 0.9)

    # surface patterns, scanned once over the corpus
    ent_keys = {key(e) for e in ents}
    pat_in = re.compile(r"([A-Z][\w'\-]+(?:[ \-][A-Z][\w'\-]+){0,3})\s+(?:in|at|near|on)\s+([A-Z][\w'\-]+(?:[ \-][A-Z][\w'\-]+){0,3})")
    # No re.I here, deliberately. Under IGNORECASE the [A-Z] entity anchor matches
    # every word, so combined with the lazy gap and {0,3} repetition the pattern
    # backtracks catastrophically over 1,200-character itineraries (observed: no
    # progress in 5 minutes at 100% CPU). Case-sensitive is also semantically
    # right, since we only want capitalised entity mentions.
    ACT_ALT = "|".join(sorted((re.escape(a) for a in ACTIVITY_WORDS), key=len, reverse=True))
    TRV_ALT = "|".join(sorted((re.escape(t) for t in TRAVELER_WORDS), key=len, reverse=True))
    pat_act = re.compile(r"([A-Z][\w'\-]+(?:[ \-][A-Z][\w'\-]+){0,3})[^.]{0,30}?\b(" + ACT_ALT + r")\b")
    pat_for = re.compile(r"([A-Z][\w'\-]+(?:[ \-][A-Z][\w'\-]+){0,3})[^.]{0,30}?\bfor\s+(" + TRV_ALT + r")\b")

    loc_c, act_c, for_c = Counter(), Counter(), Counter()
    import time as _time
    _t0 = _time.time()
    for _i, text in enumerate(records):
        if _i and _i % 5000 == 0:
            print(f"    pattern scan {_i:,}/{len(records):,} ({_time.time()-_t0:.0f}s)", flush=True)
        for a, b in pat_in.findall(text):
            ka, kb = key(norm(a)), key(norm(b))
            if ka in ent_keys and kb in ent_keys and ka != kb:
                loc_c[(canon[ka], canon[kb])] += 1
        for a, act in pat_act.findall(text):
            ka = key(norm(a))
            if ka in ent_keys:
                act_c[(canon[ka], act.lower())] += 1
        for a, tv in pat_for.findall(text):
            ka = key(norm(a))
            if ka in ent_keys:
                for_c[(canon[ka], tv.lower())] += 1

    for (a, b), c in loc_c.items():
        if c >= 4:
            add(a, "located_in", b, min(0.95, 0.5 + c / 40))
    for (a, act), c in act_c.items():
        if c >= 5:
            add(a, "has_activity", act, min(0.95, 0.5 + c / 50))
    for (a, tv), c in for_c.items():
        if c >= 4:
            add(a, "suitable_for", tv, min(0.95, 0.5 + c / 40))
    print(f"  pattern relations: located_in={sum(1 for r in rels if r['predicate']=='located_in'):,} "
          f"has_activity={sum(1 for r in rels if r['predicate']=='has_activity'):,} "
          f"suitable_for={sum(1 for r in rels if r['predicate']=='suitable_for'):,}")

    # ── 4. co-occurrence (the relation the itinerary task needs) ──────────
    pair = Counter()
    for found in per_doc:
        keep = sorted({canon[key(e)] for e in found if key(e) in canon})
        if len(keep) > 30:
            keep = keep[:30]
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                pair[(keep[i], keep[j])] += 1
    N = len(per_doc)
    cooc = defaultdict(list)
    for (a, b), c in pair.items():
        if c < MIN_PAIR_FREQ:
            continue
        pmi = math.log((c / N) / ((freq[a] / N) * (freq[b] / N)) + 1e-12)
        if pmi < MIN_PMI:
            continue
        cooc[a].append((pmi, c, b))
        cooc[b].append((pmi, c, a))
    n_cooc = 0
    for a, lst in cooc.items():
        lst.sort(reverse=True)
        for pmi, c, b in lst[:MAX_COOC_PER_ENT]:
            add(a, "co_occurs_with", b, min(0.95, 0.4 + pmi / 10))
            n_cooc += 1
    print(f"  co_occurs_with: {n_cooc:,}")

    # ── 5. carry over the curated temporal layer unchanged ────────────────
    for fn in ("temporal_facts.json", "rules.json", "schema.json"):
        src = SRC_ONTO / fn
        if src.exists():
            (DST / fn).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    (DST / "concepts.json").write_text(json.dumps(concepts, ensure_ascii=False, indent=1), encoding="utf-8")
    (DST / "relations.json").write_text(json.dumps(rels, ensure_ascii=False, indent=1), encoding="utf-8")
    hier = [{"child": c["name"], "parent": c["concept_class"]} for c in concepts
            if c["concept_class"] != "Concept"]
    (DST / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=1), encoding="utf-8")

    by_pred = Counter(r["predicate"] for r in rels)
    summary = {"version": "3.0.0", "source": "train split only (36,847 answers)",
               "concepts": len(concepts), "relations": len(rels),
               "typed_concepts": classed, "by_predicate": dict(by_pred),
               "prev_version": {"concepts": 1573, "relations": 9208}}
    (OUT / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"ontology v3.0.0 -> {DST}")
    print(f"  concepts {len(concepts):,} (was 1,573)   relations {len(rels):,} (was 9,208)")
    for p, c in by_pred.most_common():
        print(f"    {p:18s} {c:7,}")


if __name__ == "__main__":
    main()
