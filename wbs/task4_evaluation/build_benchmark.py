"""
WBS Task 4.1.1 - Benchmark construction for the six tourism task types named in
the WBS: recommendation, itinerary, safety, dining, accessibility, transport.

Built from the project's own held-out data, not written by hand and not simulated:

  source A  splits/test.jsonl       4,607 real instruction/answer pairs with gold
                                    answers. Each is routed to a category by an
                                    intent classifier whose rules are listed below,
                                    and only confidently-routed items are kept.
  source B  jeju_bench_200.jsonl    200 gazetteer-grounded factual questions whose
                                    answers are computed from coordinates, so they
                                    are auditable rather than crowd-written.

Routing is deliberately precision-oriented: an item matching two categories, or
none, is discarded rather than guessed at. The per-category counts that result are
reported honestly, including where the corpus simply has little of a category --
accessibility in particular is scarce, which is itself a finding, since the WBS
names it as a target task type while the data barely supports it.

  python -m wbs.task4_evaluation.build_benchmark --per-category 60
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Each category: (must-match any of these, veto if any of these also match).
# Vetoes keep the categories disjoint where the vocabulary overlaps.
CATEGORIES = {
    "recommendation": (
        r"\b(recommend|suggest|best|top \d|must[- ]see|worth visiting|what should i (see|visit)|"
        r"where should i go|favou?rite|highlight)\b",
        r"\b(itinerary|day trip|\d[- ]day|restaurant|eat|dish|food|bus|taxi|rental|wheelchair)\b"),
    "itinerary": (
        r"\b(itinerary|\d[- ]?day|day \d|plan (a|my|our) (trip|visit|day)|schedule|route for|"
        r"spend (a|two|three|\d+) days?)\b",
        r"\b(wheelchair|accessib)\b"),
    "safety": (
        r"\b(safe|safety|emergency|danger|hazard|jellyfish|typhoon|riptide|first aid|"
        r"police|hospital|ambulance|insurance|warning|precaution)\b",
        r""),
    "dining": (
        r"\b(restaurant|eat|dining|dish|cuisine|food|seafood|black pork|haemul|meal|"
        r"vegetarian|vegan|halal|cafe|café|breakfast menu|local speciality|specialty)\b",
        r"\b(itinerary|\d[- ]day)\b"),
    "accessibility": (
        r"\b(wheelchair|accessib|barrier[- ]free|step[- ]free|elderly|senior|stroller|"
        r"mobility|disabled|limited walking|minimal walking)\b",
        r""),
    "transport": (
        r"\b(bus|taxi|rental car|rent a car|drive|driving|airport transfer|how (do|can) i get|"
        r"transport|transit|ferry|getting around|shuttle|public transport)\b",
        r"\b(itinerary)\b"),
}


def route(text: str) -> str | None:
    t = (text or "").lower()
    hits = []
    for cat, (inc, veto) in CATEGORIES.items():
        if re.search(inc, t) and not (veto and re.search(veto, t)):
            hits.append(cat)
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pools: dict[str, list] = defaultdict(list)
    ambiguous = unrouted = 0

    test_p = ROOT / "data" / "jeju_full" / "splits" / "test.jsonl"
    for line in open(test_p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = route(r.get("Input", ""))
        if c is None:
            t = (r.get("Input") or "").lower()
            if sum(1 for cat, (inc, veto) in CATEGORIES.items()
                   if re.search(inc, t)) > 1:
                ambiguous += 1
            else:
                unrouted += 1
            continue
        pools[c].append({"source": "test_split", "category": c,
                         "record_id": r.get("record_id"),
                         "Input": r["Input"], "Output": r["Output"]})

    bench_p = ROOT / "experiments" / "exp025_multihop" / "outputs" / "jeju_bench_200.jsonl"
    n_bench = 0
    if bench_p.exists():
        for line in open(bench_p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = route(r.get("Input", ""))
            if c is None:
                continue
            n_bench += 1
            pools[c].append({"source": "gazetteer_bench", "category": c,
                             "record_id": r.get("record_id"),
                             "Input": r["Input"], "Output": r["Output"],
                             "hops": r.get("hops"), "template": r.get("template")})

    print(f"routed from test split: {sum(len(v) for v in pools.values()) - n_bench:,}")
    print(f"routed from gazetteer bench: {n_bench:,}")
    print(f"discarded: {ambiguous:,} ambiguous (multi-category), {unrouted:,} unmatched\n")

    out, stats = [], {}
    for cat in CATEGORIES:
        pool = pools.get(cat, [])
        rng.shuffle(pool)
        take = pool[: a.per_category]
        out += take
        stats[cat] = {"available": len(pool), "selected": len(take),
                      "short_by": max(0, a.per_category - len(take))}
        flag = "  << corpus-limited" if len(take) < a.per_category else ""
        print(f"  {cat:16s} available {len(pool):5,}   selected {len(take):4,}{flag}")

    rng.shuffle(out)
    for i, r in enumerate(out):
        r["bench_id"] = f"B{i+1:04d}"

    D = ROOT / "wbs" / "deliverables" / "task4_evaluation"
    D.mkdir(parents=True, exist_ok=True)
    p = D / "tourism_benchmark_categorized.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"total": len(out), "per_category_target": a.per_category, "seed": a.seed,
            "categories": stats, "discarded_ambiguous": ambiguous,
            "discarded_unmatched": unrouted,
            "sources": {"test_split": sum(1 for r in out if r["source"] == "test_split"),
                        "gazetteer_bench": sum(1 for r in out if r["source"] == "gazetteer_bench")},
            "routing_rules": {k: v[0] for k, v in CATEGORIES.items()}}
    (D / "benchmark_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"\nbenchmark: {len(out):,} items -> {p.relative_to(ROOT)}")
    short = [c for c, s in stats.items() if s["short_by"]]
    if short:
        print(f"NOTE: corpus-limited categories (fewer items than requested): {', '.join(short)}")


if __name__ == "__main__":
    main()
