"""
Stage 1 — candidate fact pool for the 200-item Jeju multi-hop benchmark.

Fairness constraint that drives the whole design: every entity used must be
attested in BOTH the ontology AND the RAG document corpus (the 36,847 train
answers). A chain grounded only in the ontology would let the ontology arm win by
construction; one grounded only in the corpus would hand it to RAG. Only the
intersection is neutral.

Jeju-only is enforced three ways: an anchor entity from the curated gazetteer,
`src.utils.jeju_guard` on the surface form, and an explicit non-Jeju blocklist.

Emits outputs/pool_multihop.jsonl and outputs/pool_single.jsonl.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.jeju_guard import JEJU_TERMS, NON_JEJU_PLACES, jeju_guard
from src.utils.proctitle import set_proctitle

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
ONTO = ROOT / "artifacts" / "ontologies" / "1.2.0"
CORPUS = ROOT / "data" / "jeju_full" / "splits" / "train.jsonl"
GAZ = ROOT / "data" / "jeju_full" / "jeju_gazetteer.json"
SEED = 20260911

# hub nodes carry no information as a bridge ("X located_in Jeju, Jeju has_activity Y")
HUBS = {"jeju", "jeju island", "jeju-do", "destination", "island", "korea",
        "south korea", "cuisine", "attraction", "activity", "tourist",
        "traveler", "accommodation", "restaurant", "trail", "region",
        "emergencyservice", "transportmode", "itinerary", "day", "day 1",
        "day 2", "day 3", "day 4", "day 5", "morning", "afternoon", "evening"}
# a chain is only "multi-hop" if the two hops say different things
TRIVIAL_PAIRS = {("near", "near"), ("located_in", "located_in"),
                 ("connected_to", "connected_to"), ("is_a", "is_a"),
                 ("part_of", "part_of")}
INFORMATIVE = {"located_in", "has_activity", "best_for", "available_in",
               "served_at", "reachable_by", "suitable_for", "requires",
               "part_of", "near", "offers", "has_seasonality", "is_a"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main() -> None:
    set_proctitle()
    rng = random.Random(SEED)

    gaz = json.loads(GAZ.read_text(encoding="utf-8"))
    anchors = set()
    for key in ("locations", "cuisine", "activities"):
        for x in gaz.get(key, []):
            name = x["name"] if isinstance(x, dict) else x
            anchors.add(norm(name))
            anchors.add(norm(re.sub(r"\s*\(.*?\)", "", name)))   # drop romanisation
    anchors.discard("")
    print(f"gazetteer anchors: {len(anchors)}")

    # ── corpus attestation: lowercase blob of all train answers ──────────────
    blob_parts = []
    for line in open(CORPUS, encoding="utf-8"):
        r = json.loads(line)
        blob_parts.append(r["Input"])
        blob_parts.append(r["Output"])
    blob = norm(" | ".join(blob_parts))
    print(f"corpus blob: {len(blob):,} chars")

    # Entity-level attestation is too weak: it admits hallucinated pairings like
    # "Darangshi Oreum has_activity snorkeling" (an oreum is a volcanic cone)
    # because both strings occur SOMEWHERE in 53M chars. Fact-level attestation
    # requires the two entities to co-occur in the same passage, which both
    # filters extraction errors and makes the fact genuinely retrievable by the
    # RAG arm — the property the fairness argument rests on.
    records = []
    for line in open(CORPUS, encoding="utf-8"):
        r = json.loads(line)
        records.append(norm(r["Input"] + " " + r["Output"]))
    print(f"corpus records: {len(records):,}")

    _postings: dict[str, set] = {}

    def postings(name: str) -> set:
        n = norm(name)
        if n not in _postings:
            _postings[n] = {i for i, t in enumerate(records) if n in t} if len(n) >= 4 else set()
        return _postings[n]

    def attested(name: str) -> bool:
        return len(postings(name)) > 0

    MIN_COOC = 2

    def fact_attested(s_: str, o_: str) -> bool:
        """Both entities named in at least MIN_COOC of the same passages."""
        if len(norm(s_)) < 4 or len(norm(o_)) < 4:
            return False
        return len(postings(s_) & postings(o_)) >= MIN_COOC

    def jeju_ok(*names: str) -> bool:
        txt = " ".join(names)
        low = norm(txt)
        if any(re.search(rf"\b{re.escape(p)}\b", low) for p in NON_JEJU_PLACES):
            return False
        ok, _ = jeju_guard(txt + " in Jeju")
        return ok

    def is_hub(name: str) -> bool:
        n = norm(name)
        return n in HUBS or len(n) < 4

    rels = json.loads((ONTO / "relations.json").read_text(encoding="utf-8"))
    cons = json.loads((ONTO / "concepts.json").read_text(encoding="utf-8"))
    cls = {c["name"]: c.get("concept_class") for c in cons if c.get("name")}

    out_edges = defaultdict(list)
    clean = []
    for r in rels:
        s, p, o = str(r.get("subject", "")), str(r.get("predicate", "")), str(r.get("object", ""))
        if not s or not o or norm(s) == norm(o) or p not in INFORMATIVE:
            continue
        clean.append((s, p, o))
        out_edges[s].append((p, o))
    print(f"usable relations: {len(clean):,}")

    attest_cache: dict[str, bool] = {}

    def ok(name: str) -> bool:
        n = norm(name)
        if n not in attest_cache:
            attest_cache[n] = attested(name)
        return attest_cache[n]

    # ── single-hop pool ──────────────────────────────────────────────────────
    single = []
    for s, p, o in clean:
        if is_hub(s) or is_hub(o):
            continue
        if norm(s) not in anchors and norm(o) not in anchors:
            continue                       # needs a real Jeju anchor
        if not fact_attested(s, o) or not jeju_ok(s, o):
            continue
        single.append({"hops": 1, "anchor": s,
                       "chain": [{"s": s, "p": p, "o": o}],
                       "classes": [cls.get(s), cls.get(o)]})

    # ── multi-hop pool: A -p1-> B -p2-> C ────────────────────────────────────
    multi, seen = [], set()
    subjects = list(out_edges)
    rng.shuffle(subjects)
    for a in subjects:
        if is_hub(a) or norm(a) not in anchors or not ok(a):
            continue
        for p1, b in out_edges[a]:
            if is_hub(b) or not fact_attested(a, b):
                continue
            for p2, c in out_edges.get(b, []):
                if is_hub(c) or norm(c) in (norm(a), norm(b)):
                    continue
                if (p1, p2) in TRIVIAL_PAIRS:
                    continue
                if not fact_attested(b, c):
                    continue
                if not jeju_ok(a, b, c):
                    continue
                key = (norm(a), norm(b), norm(c))
                if key in seen:
                    continue
                seen.add(key)
                multi.append({"hops": 2, "anchor": a,
                              "chain": [{"s": a, "p": p1, "o": b},
                                        {"s": b, "p": p2, "o": c}],
                              "classes": [cls.get(a), cls.get(b), cls.get(c)]})

    # de-duplicate single-hop by (subject, predicate) so one entity can't flood
    rng.shuffle(single)
    by_sp, single_d = defaultdict(int), []
    for x in single:
        k = (norm(x["chain"][0]["s"]), x["chain"][0]["p"])
        if by_sp[k] >= 1:
            continue
        by_sp[k] += 1
        single_d.append(x)
    rng.shuffle(multi)
    by_a, multi_d = defaultdict(int), []
    for x in multi:
        if by_a[norm(x["anchor"])] >= 3:      # cap per anchor for diversity
            continue
        by_a[norm(x["anchor"])] += 1
        multi_d.append(x)

    for name, data in (("pool_multihop", multi_d), ("pool_single", single_d)):
        p = OUT / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data):,} candidates -> {p.name}")
        for x in data[:3]:
            print("   ", " | ".join(f"{h['s']} -{h['p']}-> {h['o']}" for h in x["chain"]))


if __name__ == "__main__":
    main()
