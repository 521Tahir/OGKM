"""WBS Task 3.2.2 / proposal 10.3 - do the four constraint families fire?

10.3 asks for type, cardinality, exclusion and causal/temporal constraints. Three
of them were implemented and two were not; claiming the other two now requires
showing them run against real generated answers rather than against a unit test,
because a constraint that never fires on real output is a constraint in name only.

This replays every stored variant answer through `validate()` and reports, per
family, how many answers and how many spans it fires on, with examples. It reads
the stored JSONL files and writes a deliverable; it generates nothing and changes
no published score (`score_variants` computes its rates from `audit_spans` and the
UNSUPPORTABLE patterns, not from `validate`).

  python -m wbs.task3_reasoning.constraint_coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
OUT = ROOT / "wbs" / "deliverables" / "task3_reasoning"

FAMILY = {
    "UNGROUNDED_ENTITY": "type",
    "RELATION_UNSUPPORTED": "type",
    "UNSUPPORTABLE_CLAIM": "exclusion",
    "CARDINALITY_VIOLATION": "cardinality",
    "TEMPORAL_CONFLICT": "causal/temporal",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()

    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase, validate

    kb = KnowledgeBase()
    files = sorted(EVAL.glob(f"variant__{a.model}__*.jsonl"))
    if not files:
        sys.exit(f"no stored answers for {a.model}")

    OUT.mkdir(parents=True, exist_ok=True)
    per_arm: dict[str, dict] = {}
    ex: dict[str, list] = defaultdict(list)
    for f in files:
        arm = f.stem.split("__", 2)[2]
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        spans, answers = Counter(), Counter()
        for r in rows:
            ans = r.get("answer") or ""
            vs = validate(ans, kb)
            hit = {v["type"] for v in vs}
            for t in hit:
                answers[t] += 1
            for v in vs:
                spans[v["type"]] += 1
                if len(ex[v["type"]]) < a.examples:
                    ex[v["type"]].append({"arm": arm, "bench_id": r.get("bench_id"),
                                          "span": v["span"], "detail": v["detail"]})
        per_arm[arm] = {"n": len(rows),
                        "answers_with": dict(answers), "spans": dict(spans)}
        print(f"  {arm:<34} " + "  ".join(
            f"{t.split('_')[0][:5]} {answers.get(t, 0):3d}" for t in FAMILY), flush=True)

    fam_ans = Counter()
    for arm, d in per_arm.items():
        for t, n in d["answers_with"].items():
            fam_ans[FAMILY.get(t, "other")] += n

    L = ["# Constraint families: do they fire on real answers?\n"]
    L.append(f"Proposal 10.3 names four families. Every stored answer for "
             f"`{a.model}` is replayed through the validator; a family that never "
             f"fires is reported as never firing.\n")
    L.append(f"`is_a` is treated as functional because this ontology uses it that "
             f"way - {kb.functional.get('is_a', 0)*100:.1f}% of its subjects carry "
             f"exactly one object - and the temporal layer supplies "
             f"{len(kb.temporal)} month-scoped subjects.\n")
    L.append("| Arm | n | " + " | ".join(
        f"{t} ({FAMILY[t]})" for t in FAMILY) + " |")
    L.append("|---|--:|" + "--:|" * len(FAMILY))
    for arm, d in sorted(per_arm.items()):
        L.append(f"| {arm} | {d['n']} | " + " | ".join(
            str(d["answers_with"].get(t, 0)) for t in FAMILY) + " |")
    L.append("\nCounts are answers in which the family fires at least once.\n")

    L.append("## Per family\n")
    for t, fam in FAMILY.items():
        tot = sum(d["spans"].get(t, 0) for d in per_arm.values())
        ans = sum(d["answers_with"].get(t, 0) for d in per_arm.values())
        L.append(f"**{t}** ({fam}) - {tot:,} spans across {ans:,} answers."
                 + ("" if tot else " **Never fires on real output.**"))
        for e in ex.get(t, [])[:a.examples]:
            L.append(f"- `{e['arm']}` / {e['bench_id']}: \"{e['span']}\" — "
                     f"{e['detail']}")
        L.append("")

    missing = [f for f in set(FAMILY.values())
               if not sum(d["spans"].get(t, 0) for d in per_arm.values()
                          for t in FAMILY if FAMILY[t] == f)]
    L.append("## Reading\n")
    L.append(f"{len(set(FAMILY.values())) - len(missing)} of "
             f"{len(set(FAMILY.values()))} families fire on real generated output"
             + (f"; {', '.join(sorted(missing))} does not, and is implemented but "
                f"unexercised by this benchmark.\n" if missing else
                ", so all four are exercised rather than merely implemented.\n"))

    # where a family fires is itself a result
    for t in ("TEMPORAL_CONFLICT", "CARDINALITY_VIOLATION"):
        hits = {arm: d["answers_with"].get(t, 0) for arm, d in per_arm.items()
                if d["answers_with"].get(t, 0)}
        if not hits:
            continue
        grounded = {a_ for a_ in per_arm if not a_.startswith("V1")}
        if not (set(hits) & grounded):
            L.append(f"**{t} fires only in the ungrounded arm.** "
                     + ", ".join(f"{k} {v}" for k, v in sorted(hits.items()))
                     + f", and 0 in all {len(grounded)} grounded arms. The "
                       f"constraint is checked identically everywhere, so this is "
                       f"a property of the answers rather than of the checker: "
                       f"grounding of any kind - dense passages or ontology facts "
                       f"- removes the contradictions this family detects.\n")
    (OUT / "constraint_coverage.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "constraint_coverage.json").write_text(json.dumps(
        {"model": a.model, "families": FAMILY,
         "functional_predicates": kb.functional,
         "temporal_subjects": len(kb.temporal),
         "per_arm": per_arm, "examples": {k: v for k, v in ex.items()}},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'constraint_coverage.md'}")
    for fam, n in sorted(fam_ans.items()):
        print(f"  {fam:<16} {n:,} answers")


if __name__ == "__main__":
    main()
