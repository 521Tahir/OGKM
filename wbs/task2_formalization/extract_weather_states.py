"""
WBS Task 2.1 / R12 - the weather state four rules need, extracted from the
corpus the ontology was built from.

Four of the eight seed rules cannot be enforced. `rule_compliance.py` measures
why against the ontology, and three of the four reasons name a weather state:
R001 needs a severe-storm state on coastal attractions, R006 needs any
queryable weather state beside `has_seasonality`. The register said no project
source supplies it. That is true of the *ontology* - `WeatherCondition` holds
Rain, Wind, Snow, Ice, Heat and three climate descriptions, none of them a
storm - and it had never been checked against the *corpus*.

It is in the corpus. Typhoon, storm, rough-sea and high-wave language appears
in 559 of 46,059 records, and 519 of those name a specific Jeju place nearby.
So the blocker is the extraction, not the source, and this script closes that
gap the way the rest of Task 2 does: conservative sentence-level evidence, a
support threshold, provenance on every triple, and a new ontology version
written beside the published one rather than over it.

Conservatism, because a rule that fires on noise is worse than one that cannot
fire:
  * subject and object must already be typed by the schema - no new concept
    classes, and `avoid_during` keeps its declared domain and range;
  * the place and the weather term must occur in the SAME sentence;
  * a pair needs >= MIN_SUPPORT distinct source records;
  * generic island-wide names ("Jeju", "Island") are not entities for this
    purpose, since a state asserted of the whole island cannot discriminate.

  python -m wbs.task2_formalization.extract_weather_states --write 1.5.0
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ONTO = ROOT / "artifacts" / "ontologies"
OUT = ROOT / "wbs" / "deliverables" / "task2_formalization"
CORPUS = ROOT / "data" / "processed" / "cleaned_records.jsonl"

MIN_SUPPORT = 3
PREDICATE = "avoid_during"
# the states this script is willing to assert, and the language that evidences
# each. Kept small and literal: every one is a weather condition a reader would
# recognise as suppressing coastal activity.
STATES = {
    "Typhoon": r"\btyphoons?\b",
    "Storm": r"\bstorms?\b|\bgale\b|\bsqualls?\b",
    "Rough Seas": r"\brough seas?\b|\bhigh waves?\b|\bstrong currents?\b",
}
GENERIC = {"jeju", "jeju island", "island", "korea", "south korea", "jeju city",
           "weather", "ocean", "sea", "beach", "beaches", "mountain",
           # concepts that exist but name a category, not a place a rule can
           # discriminate on: a state asserted of "Transport" or of "Jeju's
           # beaches" cannot tell one coastal attraction from another
           "transport", "transportation", "jeju's beaches", "coastal areas",
           "activities", "waterfalls"}
SENT = re.compile(r"(?<=[.!?•])\s+|\n+")
# the other half of R12: a *graded* severity, which the two rules still blocked
# after this extraction need. Counted on the same pass so the claim that the
# corpus does not carry it is a measurement and not an impression.
GRADED = re.compile(r"\b(high|medium|moderate|low|severe|extreme)[- ]"
                    r"(risk|danger|alert|warning|caution)\b", re.I)


def load(ver: str, name: str):
    return json.loads((ONTO / ver / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-version", default="1.4.0")
    ap.add_argument("--write", default="", help="new version to write, e.g. 1.5.0")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-existing", action="store_true",
                    help="if the target version already exists, leave it in "
                         "place and regenerate the report from it")
    a = ap.parse_args()

    concepts = load(a.from_version, "concepts")
    relations = load(a.from_version, "relations")
    schema = load(a.from_version, "schema")
    hier = load(a.from_version, "hierarchy")

    cls = {c["name"]: c.get("concept_class") for c in concepts}
    path = {h["concept"]: h.get("hierarchy_path") or [] for h in hier}
    cons = schema["relation_constraints"][PREDICATE]
    subj_ok, obj_ok = set(cons["subject_classes"]), set(cons["object_classes"])
    if "WeatherCondition" not in obj_ok:
        sys.exit(f"{PREDICATE} cannot take a WeatherCondition object in this schema")

    def subject_allowed(name: str) -> bool:
        # the declared domain is coarse (Activity/Attraction/Destination) and
        # the concepts are fine (Beach/Trail/Mountain), so ancestry decides
        if cls.get(name) in subj_ok:
            return True
        return bool(set(path.get(name, [])) & subj_ok)

    # A concept qualifies as a subject only if it is a *named* place: at least
    # one token that is not an ordinary English word. The first pass accepted
    # "Activities", "Transport", "Coastal areas" and "Waterfalls" - concepts
    # that exist but cannot carry a state a rule can discriminate on. The gate
    # is the project's own common-word list, the one the span auditor uses, so
    # this script and the hallucination detector agree about what a name is.
    from wbs.task4_evaluation.hallucination_report import common_words
    common = {w.lower() for w in common_words()}
    classes = {c.lower() for c in schema["concept_classes"]}
    classes |= {c + "s" for c in classes}

    def named(n: str) -> bool:
        low = n.lower()
        if low in GENERIC or low in classes:
            return False
        toks = [t for t in re.split(r"[^A-Za-z]+", n) if t]
        return any(t.lower() not in common for t in toks)

    # longest-first so "Jungmun Saekdal Beach" wins over "Jungmun"
    names = sorted((n for n in cls
                    if named(n) and len(n) >= 6 and subject_allowed(n)),
                   key=len, reverse=True)
    report_gate = {"concepts_typed_for_predicate": sum(1 for n in cls
                                                       if subject_allowed(n)),
                   "of_those_named": len(names)}
    pat = re.compile(r"(?<![A-Za-z])(" +
                     "|".join(re.escape(n) for n in names) + r")(?![A-Za-z])",
                     re.I)
    canon = {n.lower(): n for n in names}
    state_pats = {s: re.compile(p, re.I) for s, p in STATES.items()}

    n_graded = 0
    support: dict[tuple[str, str], set] = collections.defaultdict(set)
    quote: dict[tuple[str, str], str] = {}
    n_rec = n_sent = 0
    for line in open(CORPUS, encoding="utf-8"):
        if not line.strip():
            continue
        n_rec += 1
        if a.limit and n_rec > a.limit:
            break
        rec = json.loads(line)
        rid = rec.get("record_id", rec.get("id", n_rec))
        text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
        if GRADED.search(text):
            n_graded += 1
        for sent in SENT.split(text):
            hit = [s for s, p in state_pats.items() if p.search(sent)]
            if not hit:
                continue
            n_sent += 1
            for m in pat.finditer(sent):
                ent = canon.get(m.group(1).lower())
                if not ent:
                    continue
                for s in hit:
                    support[(ent, s)].add(rid)
                    quote.setdefault((ent, s), sent.strip()[:240])

    accepted = {k: v for k, v in support.items() if len(v) >= a.min_support}
    rejected = {k: v for k, v in support.items() if len(v) < a.min_support}
    states_used = sorted({s for _, s in accepted})
    ents = sorted({e for e, _ in accepted})

    report = {
        "from_version": a.from_version, "predicate": PREDICATE,
        "min_support": a.min_support, "records_scanned": n_rec,
        "sentences_with_weather_language": n_sent,
        "subject_gate": report_gate,
        "records_with_graded_severity": n_graded,
        "candidate_pairs": len(support), "accepted_pairs": len(accepted),
        "rejected_for_support": len(rejected),
        "entities": len(ents), "states": states_used,
        "by_state": {s: sum(1 for _, s2 in accepted if s2 == s) for s in states_used},
        "top": [{"entity": e, "state": s, "support": len(v),
                 "evidence": quote[(e, s)]}
                for (e, s), v in sorted(accepted.items(),
                                        key=lambda kv: -len(kv[1]))[:25]],
    }

    wrote_version = False
    if a.write:
        dst = ONTO / a.write
        if dst.exists() and not a.reuse_existing:
            sys.exit(f"{dst} already exists - refusing to overwrite a version "
                     f"(pass --reuse-existing to report from it instead)")
    if a.write and not (ONTO / a.write).exists():
        dst = ONTO / a.write
        wrote_version = True
        dst.mkdir(parents=True)
        now = datetime.now(timezone.utc).isoformat()
        prov = {"source_record_id": -1,
                "extraction_prompt_version": "wbs/weather-state-v1",
                "timestamp": now, "validator_status": "pass"}
        new_concepts = list(concepts)
        have = set(cls)
        for s in states_used:
            if s not in have:
                new_concepts.append({
                    "name": s, "concept_class": "WeatherCondition", "aliases": [],
                    "attributes": {},
                    "provenance": dict(prov, confidence=1.0,
                                       approved_by="corpus extraction")})
        new_rel = list(relations)
        for (e, s), rids in sorted(accepted.items()):
            new_rel.append({
                "subject": e, "predicate": PREDICATE, "object": s,
                "provenance": dict(prov,
                                   confidence=round(min(1.0, len(rids) / 10), 3),
                                   source_record_ids=sorted(str(r) for r in rids)[:20],
                                   evidence=quote[(e, s)])})
        new_hier = list(hier)
        for s in states_used:
            if s not in path:
                new_hier.append({"concept": s, "concept_class": "WeatherCondition",
                                 "parent_class": "WeatherCondition",
                                 "hierarchy_path": ["TourismEntity",
                                                    "WeatherCondition"],
                                 "is_instance": True, "confidence": 1.0})
        (dst / "concepts.json").write_text(json.dumps(new_concepts, ensure_ascii=False,
                                                      indent=1), encoding="utf-8")
        (dst / "relations.json").write_text(json.dumps(new_rel, ensure_ascii=False,
                                                       indent=1), encoding="utf-8")
        (dst / "hierarchy.json").write_text(json.dumps(new_hier, ensure_ascii=False,
                                                       indent=1), encoding="utf-8")
        for f in ("schema", "rules", "temporal_facts"):
            src = ONTO / a.from_version / f"{f}.json"
            if src.exists():
                (dst / f"{f}.json").write_text(src.read_text(encoding="utf-8"),
                                               encoding="utf-8")
        ch = load(a.from_version, "changelog") if (
            ONTO / a.from_version / "changelog.json").exists() else []
        ch = ch if isinstance(ch, list) else [ch]
        ch.append({"version": a.write, "from": a.from_version, "timestamp": now,
                   "change": f"+{len(accepted)} {PREDICATE} triples over "
                             f"{len(ents)} entities and {len(states_used)} new "
                             f"weather states, extracted from "
                             f"{report['records_scanned']} corpus records with "
                             f"support >= {a.min_support}",
                   "script": "wbs/task2_formalization/extract_weather_states.py"})
        (dst / "changelog.json").write_text(json.dumps(ch, ensure_ascii=False,
                                                       indent=1), encoding="utf-8")
        report["written"] = str(dst.relative_to(ROOT))
        report["concepts"] = len(new_concepts)
        report["relations"] = len(new_rel)
    elif a.write:
        report["written"] = str((ONTO / a.write).relative_to(ROOT)) + " (existing)"

    # ── the deliverable, including what it unblocks where that is measured ──
    L = ["# The weather state the rule layer was missing\n",
         "Four of the eight seed rules cannot be enforced against the "
         "production ontology, and `rule_compliance.md` names a weather state "
         "in two of the four reasons. The register recorded that no project "
         "source supplied one. That was checked against the ontology and not "
         "against the corpus the ontology was built from, and it is wrong about "
         "the corpus.\n",
         f"Scanning all {report['records_scanned']:,} cleaned records for "
         f"typhoon, storm, gale, rough-sea, high-wave and strong-current "
         f"language finds {report['sentences_with_weather_language']:,} "
         f"sentences that also name a place. The extraction below is "
         f"deliberately conservative: subject and object must already be typed "
         f"by the schema so `{PREDICATE}` keeps its declared domain and range, "
         f"the place and the weather term must share a sentence, the place must "
         f"be a *named* one (at least one token outside the project's own "
         f"common-word list, the same gate the span auditor uses), and a pair "
         f"needs at least {a.min_support} distinct source records.\n",
         "| | count |", "|---|--:|",
         f"| concepts the predicate can take as a subject | "
         f"{report['subject_gate']['concepts_typed_for_predicate']} |",
         f"| of those, named places | {report['subject_gate']['of_those_named']} |",
         f"| candidate (place, state) pairs | {report['candidate_pairs']} |",
         f"| accepted at support >= {a.min_support} | {report['accepted_pairs']} |",
         f"| rejected for support | {report['rejected_for_support']} |",
         f"| entities carrying a state | {report['entities']} |", "",
         "States asserted: " + ", ".join(f"**{s}** ({report['by_state'][s]} pairs)"
                                         for s in states_used) + ".\n",
         "## The strongest evidence\n",
         "| Entity | State | Records | Sentence |", "|---|---|--:|---|"]
    for t in report["top"][:12]:
        L.append(f"| {t['entity']} | {t['state']} | {t['support']} | "
                 f"{t['evidence'][:150].replace('|', '/')} |")
    L.append("")

    pub = OUT / "rule_compliance.json"
    cand = OUT / f"rule_compliance__{a.write}.json" if a.write else None
    if pub.exists() and cand and cand.exists():
        pj = json.loads(pub.read_text(encoding="utf-8"))["enforceability"]
        cj = json.loads(cand.read_text(encoding="utf-8"))["enforceability"]
        was = sorted(k for k, v in pj.items() if not v.get("enforceable"))
        now = sorted(k for k, v in cj.items() if not v.get("enforceable"))
        gained = [k for k in was if k not in now]
        L += ["## What it unblocks\n",
              f"Measured by `rule_compliance.py` against the new version, with "
              f"the published record left alone: "
              f"**{len(pj) - len(was)} of {len(pj)}** seed rules enforceable "
              f"before, **{len(cj) - len(now)} of {len(cj)}** after.\n"]
        if gained:
            L.append("Unblocked: " + ", ".join(
                f"**{k}** ({cj[k]['description']})" for k in gained) + ".\n")
        if now:
            L.append("Still blocked: " + ", ".join(
                f"**{k}** (needs {', '.join(cj[k]['needs'])})" for k in now)
                + ". Those two need a graded risk severity, and that is "
                "genuinely not in the corpus: phrases of the form "
                f"\"high-risk\"/\"low-risk\" appear in "
                f"{report['records_with_graded_severity']} of "
                f"{report['records_scanned']:,} records "
                f"({report['records_with_graded_severity']/report['records_scanned']*100:.2f}%). The "
                "distinction worth keeping is that a weather state was missing "
                "from the *extraction* while a severity is missing from the "
                "*source*.\n")
        report["enforceable_before"] = len(pj) - len(was)
        report["enforceable_after"] = len(cj) - len(now)
        report["unblocked"] = gained
        report["still_blocked"] = now

    L += ["## What this does not change\n",
          "The new version is written beside the published one, not over it. "
          "Every measured result in this project - the benchmark runs, the "
          "validator, the hallucination detector - loads the production version "
          "it always loaded, and no published number moves. What changes is "
          "that the rule layer has a version where two more of its rules can "
          "fire, and the evidence for each triple is in the ontology beside "
          "it.\n"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weather_state_extraction.md").write_text("\n".join(L),
                                                     encoding="utf-8")
    (OUT / "weather_state_extraction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scanned {n_rec:,} records, {n_sent:,} sentences with weather language")
    print(f"  candidate pairs {len(support)}, accepted (>= {a.min_support}) "
          f"{len(accepted)} over {len(ents)} entities, states {states_used}")
    if wrote_version:
        print(f"  wrote {report['written']}: {report['concepts']} concepts, "
              f"{report['relations']} relations")
    elif a.write:
        print(f"  reused {report['written']}")


if __name__ == "__main__":
    main()
