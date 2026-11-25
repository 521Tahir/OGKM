"""
Build a large temporal QA benchmark (up to 1000 questions) by expanding the
curated Jeju temporal facts with question templates. Each question keeps a gold
answer + key month tokens, matching the comments.txt Case A/B setup.

NOTE: there are not 1000 *distinct* Jeju temporal facts, so beyond the ~50 seed
questions these are paraphrase variants of the ~59 underlying facts (different
phrasings of the same time-dependent fact). Transparent by design.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.io import load_json, save_json

FACTS = ROOT / "data" / "temporal" / "jeju_temporal_facts.json"
SEED50 = ROOT / "data" / "temporal" / "temporal_qa_benchmark.json"
MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

TEMPLATES = [
    "When {be} {s} in Jeju?",
    "In which months {be} {s} in Jeju?",
    "What time of year {be} {s} in Jeju?",
    "When can I experience {s} in Jeju?",
    "What months are best for {s} in Jeju?",
    "When should I visit Jeju for {s}?",
    "Which season {be} {s} in Jeju?",
    "When does {s} happen in Jeju?",
    "What is the best time for {s} in Jeju?",
    "When is the season for {s} in Jeju?",
    "During which months {be} {s} in Jeju?",
    "When {be} {s} typically seen in Jeju?",
    "What months should I plan a Jeju trip to catch {s}?",
    "When {be} {s} at its peak in Jeju?",
    "At what time of year {be} {s} in Jeju?",
    "Could you tell me when {s} {be} in Jeju?",
    "I want to see {s} in Jeju — which months?",
]


def key_tokens(months):
    ms = sorted(set(m for m in months if 1 <= m <= 12))
    if not ms:
        return []
    if len(ms) == 1:
        return [MONTHS[ms[0]].lower()]
    return [MONTHS[ms[0]].lower(), MONTHS[ms[-1]].lower()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/temporal/temporal_qa_1000.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    facts = load_json(FACTS)["facts"]
    seed_qs = load_json(SEED50)["questions"]

    questions = []
    seen = set()
    # 1) high-quality hand-written seed questions first
    for q in seed_qs:
        key = q["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            questions.append({"id": len(questions) + 1, "category": q["category"],
                              "question": q["question"], "gold_answer": q["gold_answer"],
                              "key_tokens": q["key_tokens"], "source": "seed"})
    # 2) template-expanded questions over the fact base
    pool = []
    for f in facts:
        months = f["temporal"].get("months", [])
        if not months:
            continue  # 'always'/undated facts can't make month-specific questions
        s = f["subject"]
        s_l = s[0].lower() + s[1:] if s and s[0].isupper() and not s.isupper() else s
        be = "are" if s.endswith("s") and not s.endswith("ss") else "is"
        note = f["temporal"]["note"]
        kt = key_tokens(months)
        gold = f"{s} in Jeju: {note}."
        for tpl in TEMPLATES:
            q = tpl.format(s=s_l, be=be)
            pool.append({"category": f["category"], "question": q,
                         "gold_answer": gold, "key_tokens": kt, "source": "template"})
    rng.shuffle(pool)
    for item in pool:
        if len(questions) >= args.n:
            break
        key = item["question"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        item["id"] = len(questions) + 1
        questions.append(item)

    rng.shuffle(questions)
    for i, q in enumerate(questions, 1):
        q["id"] = i

    out = {"meta": {"description": f"Temporal QA benchmark ({len(questions)} Q) — seed + template-expanded "
                    f"from {len(facts)} Jeju temporal facts. Beyond the ~50 seed, questions are paraphrase "
                    f"variants of the underlying facts.", "version": "1.2.0",
                    "n_facts": len(facts), "n_seed": len(seed_qs)},
           "questions": questions}
    save_json(out, ROOT / args.out)
    from collections import Counter
    print(f"Built {len(questions)} temporal questions → {args.out}")
    print("  by source:", dict(Counter(q["source"] for q in questions)))
    print("  by category:", dict(Counter(q["category"] for q in questions)))


if __name__ == "__main__":
    main()
