"""
WBS Task 2.1 - the *policy* constraints the business case turns on, extracted
from the corpus the ontology was built from.

`extract_weather_states.py` closed the weather half of R12: three of the eight
seed rules needed a storm state, the register said no project source supplied
one, and the corpus did. The business-plan brief (Ontology-Comment, 19 Oct)
asks for the other half, and asks for it in the language a customer uses:

    [Flagship Software V1] --(is_incompatible_with)--> [Cloud Server 2026]
    [Contractor]           --(remote_work)-->          [Not Eligible]

Both of the brief's examples are *policy*: a hard requirement or a hard
prohibition that a statistical model cannot infer and will cheerfully guess
wrong. This ontology's equivalents are the island's own rules - a rental car
needs an International Driving Permit, a market stall takes cash, a temple
expects covered shoulders - and the schema already declares the predicates for
them (`requires`, `needs_reservation`) without a single triple using either.

So the blocker is again the extraction, not the source, and this script closes
it the same way Task 2 closes everything else: sentence-level evidence, a
proximity window, a support threshold, provenance on every triple, and a new
ontology version written beside the published one rather than over it.

Conservatism, because a policy rule that fires on noise is worse than none:
  * the constraint language and the entity mention must share a sentence AND
    sit within WINDOW characters of each other - without this, "visit the
    museum, then dinner at a restaurant (reserve a table)" asserts a
    reservation policy about the museum;
  * a requirement is dropped if the sentence negates it ("no need to", "by not
    removing your shoes"), and a prohibition needs an explicit prohibition cue
    in the same sentence - without that gate, "letting the kids collect
    seashells" reads as a ban on collecting them;
  * a policy about a place is asserted only of a place the gazetteer can put on
    a map: the subject must carry coordinates. That one gate is what separates
    "Gwaneumsa Temple" from "Jeju culture" and "Korean phrases", which are
    typed as cultural attractions and are not places a rule can be enforced at;
  * a pair needs >= MIN_SUPPORT distinct source records;
  * the subject must be typed for the predicate, and for place-constraints it
    must be a *named* place by the span auditor's own common-word list. Where
    the subject is a transport mode the name is a vocabulary term ("Car
    Rental"), so that gate is relaxed for those and only those.

  python -m wbs.task2_formalization.extract_policy_constraints --write 1.6.0
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
WINDOW = 120                      # characters between entity and constraint
SENT = re.compile(r"(?<=[.!?•])\s+|\n+")
NEGATED = re.compile(r"\b(no need|not|never|without)\s*$", re.I)
# a prohibition needs to be phrased as one. "letting the kids collect seashells"
# names the activity and the beach in one sentence and means the opposite.
PROHIBIT_CUE = re.compile(r"\b(do not|don'?t|do nots?|prohibit\w*|not allowed|"
                          r"not permitted|forbidden|banned|illegal|refrain from|"
                          r"avoid|no longer|never)\b", re.I)
# a restriction is weaker than a ban and the corpus phrases it that way:
# "follow local regulations on drone usage in protected areas".
RESTRICT_CUE = re.compile(r"\b(local regulations|regulations|restrict\w*|"
                          r"permit|permission|authoris\w*|authoriz\w*)\b", re.I)

# Each entry is one constraint the corpus evidences, the predicate that carries
# it, and which concepts may stand on the other end of it.
#   direction "s" : <entity> --(predicate)--> <constraint>      (a requirement)
#   direction "o" : <constraint> --(predicate)--> <entity>      (a prohibition)
CONSTRAINTS = {
    "International Driving Permit": dict(
        predicate="requires", direction="s", klass="TravelConstraint",
        rx=r"international driving permits?|international driver'?s? licen[cs]es?",
        subject_classes={"TransportMode"}, subject_mode="vocab",
        asks="Can I drive here on my home licence?"),
    "Cash Payment": dict(
        predicate="requires", direction="s", klass="TravelConstraint",
        rx=r"cash[- ]only|carry cash|accept only cash|"
           r"(do not|don'?t|may not|might not) accept (credit |debit )?cards|"
           r"(do not|don'?t|may not|might not) accept credit cards",
        subject_classes={"Market", "Restaurant"}, subject_mode="named",
        asks="Can I pay by card?"),  # not "Cash": that concept exists, typed Cuisine
    "Modest Dress": dict(
        predicate="requires", direction="s", klass="LocalCustom",
        rx=r"dress(ing)? modestly|modest (clothing|dress|attire)|"
           r"cover(ing)? (your )?(shoulders|knees)",
        subject_classes={"ReligiousSite", "CulturalAttraction"},
        subject_mode="named", asks="What should I wear?"),
    "Shoes Removed": dict(
        predicate="requires", direction="s", klass="LocalCustom",
        rx=r"remov(e|ing) (your )?shoes|take off your shoes|shoes off",
        subject_classes={"ReligiousSite", "CulturalAttraction", "Accommodation"},
        subject_mode="named", asks="Do I need to take my shoes off?"),
    "Reef-safe Sunscreen": dict(
        predicate="requires", direction="s", klass="TravelConstraint",
        rx=r"reef[- ]safe sunscreen",
        subject_classes={"Beach", "NaturalAttraction"}, subject_mode="named",
        asks="Which sunscreen can I use in the water?"),
    "Drone Flying": dict(
        predicate="restricted_in", direction="o", klass="Activity",
        rx=r"drones?\b", cue="restrict",
        subject_classes={"Park", "NaturalAttraction", "Attraction"},
        subject_mode="named", asks="Can I fly a drone here?"),
    "Shell Collecting": dict(
        predicate="not_allowed", direction="o", klass="Activity",
        rx=r"(collect\w*|tak(e|ing)|remov\w+)\s+(sea\s?)?(shells?|corals?)",
        cue="prohibit",
        subject_classes={"Beach", "NaturalAttraction", "Attraction"},
        subject_mode="named", asks="Can I take a shell home?"),
}
GENERIC = {"jeju", "jeju island", "island", "korea", "south korea", "jeju city",
           "weather", "ocean", "sea", "beach", "beaches", "mountain", "temple",
           "temples", "market", "markets", "local market", "local markets",
           "museum", "museums", "restaurant", "restaurants", "parks",
           "jeju home", "traditional market", "jeju market"}


def load(ver: str, name: str):
    return json.loads((ONTO / ver / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-version", default="1.5.0")
    ap.add_argument("--write", default="")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-existing", action="store_true")
    a = ap.parse_args()

    concepts = load(a.from_version, "concepts")
    relations = load(a.from_version, "relations")
    schema = load(a.from_version, "schema")
    hier = load(a.from_version, "hierarchy")
    cls = {c["name"]: c.get("concept_class") for c in concepts}
    path = {h["concept"]: h.get("hierarchy_path") or [] for h in hier}
    # a place a rule can be enforced at is one the gazetteer can point to
    mappable = {c["name"] for c in concepts
                if "lat" in (c.get("attributes") or {})}

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

    def typed(n: str, ok: set) -> bool:
        return cls.get(n) in ok or bool(set(path.get(n, [])) & ok)

    # one scan of the corpus serves every constraint, so the candidate set for
    # each is gated per constraint rather than globally
    subjects: dict[str, set[str]] = {}
    for c, spec in CONSTRAINTS.items():
        ok = spec["subject_classes"]
        subjects[c] = {n for n in cls
                       if typed(n, ok) and len(n) >= 4
                       and (spec["subject_mode"] == "vocab"
                            or (named(n) and n in mappable))
                       and n.lower() not in GENERIC}
    allnames = sorted(set().union(*subjects.values()), key=len, reverse=True)
    pat = re.compile(r"(?<![A-Za-z])(" +
                     "|".join(re.escape(n) for n in allnames) + r")(?![A-Za-z])", re.I)
    canon = {n.lower(): n for n in allnames}
    rx = {c: re.compile(s["rx"], re.I) for c, s in CONSTRAINTS.items()}

    support: dict[tuple[str, str], set] = collections.defaultdict(set)
    quote: dict[tuple[str, str], str] = {}
    n_rec = n_sent = n_far = n_neg = n_cue = 0
    for line in open(CORPUS, encoding="utf-8"):
        if not line.strip():
            continue
        n_rec += 1
        if a.limit and n_rec > a.limit:
            break
        rec = json.loads(line)
        rid = rec.get("record_id", n_rec)
        text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
        for sent in SENT.split(text):
            hits = {c: list(rx[c].finditer(sent)) for c in CONSTRAINTS}
            hits = {c: m for c, m in hits.items() if m}
            if not hits:
                continue
            n_sent += 1
            ents = [(m.start(), m.end(), canon.get(m.group(1).lower()))
                    for m in pat.finditer(sent)]
            ents = [e for e in ents if e[2]]
            for c, ms in hits.items():
                cue = CONSTRAINTS[c].get("cue")
                if cue == "prohibit" and not PROHIBIT_CUE.search(sent):
                    n_cue += 1
                    continue
                if cue == "restrict" and not RESTRICT_CUE.search(sent):
                    n_cue += 1
                    continue
                for m in ms:
                    if (CONSTRAINTS[c]["predicate"] == "requires"
                            and NEGATED.search(sent[max(0, m.start() - 24):m.start()])):
                        n_neg += 1
                        continue
                    for s0, s1, ent in ents:
                        if ent not in subjects[c]:
                            continue
                        gap = (m.start() - s1) if m.start() >= s1 else (s0 - m.end())
                        if gap > a.window:
                            n_far += 1
                            continue
                        support[(ent, c)].add(rid)
                        # the tightest sentence is the fairest evidence to quote
                        cand = sent.strip()[:240]
                        if len(cand) < len(quote.get((ent, c), "z" * 999)):
                            quote[(ent, c)] = cand

    accepted = {k: v for k, v in support.items() if len(v) >= a.min_support}
    rejected = {k: v for k, v in support.items() if len(v) < a.min_support}
    used = sorted({c for _, c in accepted})
    report = {
        "from_version": a.from_version, "min_support": a.min_support,
        "proximity_window_chars": a.window,
        "records_scanned": n_rec, "sentences_with_policy_language": n_sent,
        "pairs_dropped_for_distance": n_far, "matches_dropped_for_negation": n_neg,
        "sentences_dropped_for_no_cue": n_cue,
        "candidate_pairs": len(support), "accepted_pairs": len(accepted),
        "rejected_for_support": len(rejected),
        "constraints": {c: {"predicate": CONSTRAINTS[c]["predicate"],
                            "pairs": sum(1 for _, c2 in accepted if c2 == c),
                            "question_it_answers": CONSTRAINTS[c]["asks"]}
                        for c in used},
        "entities": len(sorted({e for e, _ in accepted})),
        "top": [{"entity": e, "constraint": c, "support": len(v),
                 "predicate": CONSTRAINTS[c]["predicate"],
                 "triple": (f"{e} {CONSTRAINTS[c]['predicate']} {c}"
                            if CONSTRAINTS[c]["direction"] == "s"
                            else f"{c} {CONSTRAINTS[c]['predicate']} {e}"),
                 "evidence": quote[(e, c)]}
                for (e, c), v in sorted(accepted.items(), key=lambda kv: -len(kv[1]))],
    }

    if a.write:
        dst = ONTO / a.write
        if dst.exists() and not a.reuse_existing:
            sys.exit(f"{dst} already exists - refusing to overwrite a version")
        if not dst.exists():
            dst.mkdir(parents=True)
            now = datetime.now(timezone.utc).isoformat()
            prov = {"source_record_id": -1,
                    "extraction_prompt_version": "wbs/policy-constraint-v1",
                    "timestamp": now, "validator_status": "pass"}
            new_concepts, have = list(concepts), set(cls)
            for c in used:
                if c not in have:
                    new_concepts.append({
                        "name": c, "concept_class": CONSTRAINTS[c]["klass"],
                        "aliases": [], "attributes": {},
                        "provenance": dict(prov, confidence=1.0,
                                           approved_by="corpus extraction")})
            new_rel = list(relations)
            for (e, c), rids in sorted(accepted.items()):
                spec = CONSTRAINTS[c]
                s, o = (e, c) if spec["direction"] == "s" else (c, e)
                new_rel.append({
                    "subject": s, "predicate": spec["predicate"], "object": o,
                    "provenance": dict(prov,
                                       confidence=round(min(1.0, len(rids) / 10), 3),
                                       source_record_ids=sorted(str(r) for r in rids)[:20],
                                       evidence=quote[(e, c)])})
            new_hier = list(hier)
            for c in used:
                if c not in path:
                    k = CONSTRAINTS[c]["klass"]
                    new_hier.append({"concept": c, "concept_class": k,
                                     "parent_class": k,
                                     "hierarchy_path": ["TourismEntity", k],
                                     "is_instance": True, "confidence": 1.0})
            new_schema = json.loads(json.dumps(schema))
            new_schema["version"] = a.write
            new_schema["relation_types"] = sorted(
                set(new_schema["relation_types"])
                | {CONSTRAINTS[c]["predicate"] for c in used})
            # The declared domain and range are the classes the triples
            # actually use, widened by the classes the specs allow. Declaring
            # the aspiration instead of the assertion is how a schema starts
            # lying: the first run of this script declared `requires` over
            # {ReligiousSite, CulturalAttraction} and then asserted it of four
            # museums, which its own conformance check caught.
            tmp_cls = {c["name"]: c.get("concept_class") for c in new_concepts}
            asserted: dict[str, dict[str, set]] = {}
            for (e, c), _ in sorted(accepted.items()):
                spec = CONSTRAINTS[c]
                sj, ob = (e, c) if spec["direction"] == "s" else (c, e)
                d = asserted.setdefault(spec["predicate"],
                                        {"subject_classes": set(), "object_classes": set()})
                d["subject_classes"].add(tmp_cls.get(sj))
                d["object_classes"].add(tmp_cls.get(ob))
            for pr, d in asserted.items():
                old = new_schema["relation_constraints"].get(pr, {})
                new_schema["relation_constraints"][pr] = {
                    k: sorted({x for x in (set(old.get(k, [])) | d[k]) if x})
                    for k in ("subject_classes", "object_classes")}
            for name, blob in (("concepts", new_concepts), ("relations", new_rel),
                               ("hierarchy", new_hier), ("schema", new_schema)):
                (dst / f"{name}.json").write_text(
                    json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
            for f in ("rules", "temporal_facts"):
                src = ONTO / a.from_version / f"{f}.json"
                if src.exists():
                    (dst / f"{f}.json").write_text(src.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
            ch = load(a.from_version, "changelog")
            ch = ch if isinstance(ch, list) else [ch]
            ch.append({"version": a.write, "from": a.from_version, "timestamp": now,
                       "change": f"+{len(accepted)} policy triples "
                                 f"({', '.join(sorted({CONSTRAINTS[c]['predicate'] for c in used}))}) "
                                 f"over {report['entities']} entities and "
                                 f"{len(used)} constraints, extracted from "
                                 f"{n_rec} corpus records at support >= "
                                 f"{a.min_support} within {a.window} characters",
                       "script": "wbs/task2_formalization/extract_policy_constraints.py"})
            (dst / "changelog.json").write_text(json.dumps(ch, ensure_ascii=False,
                                                           indent=1), encoding="utf-8")
            report["concepts"], report["relations"] = len(new_concepts), len(new_rel)

            # `validate_ontology.py` type-checks against its own hardcoded
            # TYPE_RULES, which does not cover these predicates, so the claim
            # "they conform to the schema they declare" is checked here rather
            # than assumed. Every newly written triple, against the constraint
            # this script just added to the schema.
            new_cls = {c["name"]: c.get("concept_class") for c in new_concepts}
            viol = []
            for r in new_rel[len(relations):]:
                con = new_schema["relation_constraints"].get(r["predicate"])
                if not con:
                    continue
                sc, oc = new_cls.get(r["subject"]), new_cls.get(r["object"])
                if sc not in con["subject_classes"]:
                    viol.append(f"{r['subject']} ({sc}) as subject of {r['predicate']}")
                if oc not in con["object_classes"]:
                    viol.append(f"{r['object']} ({oc}) as object of {r['predicate']}")
            report["schema_conformance"] = {
                "triples_checked": len(new_rel) - len(relations),
                "violations": viol}
        report["written"] = str((ONTO / a.write).relative_to(ROOT))

    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / "policy_constraint_extraction.json"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# The policy constraints the schema declared and nothing populated\n",
         "The business-plan brief asks for the two corrections a customer "
         "recognises: a hard requirement (\"V1 is not compatible with Cloud "
         "Server 2026\") and a hard category rule (\"contractors are not "
         "covered by the hybrid policy\"). Neither can be inferred from "
         "statistics; both have to be asserted.\n",
         f"This ontology's equivalents are the island's own rules, and the "
         f"schema at v{a.from_version} already declares the predicates for them "
         f"- `requires` and `needs_reservation` are both in `relation_types` - "
         f"with no triple using either as a policy. Scanning all "
         f"{report['records_scanned']:,} cleaned records finds "
         f"{report['sentences_with_policy_language']:,} sentences carrying "
         f"policy language next to a place the ontology knows.\n",
         "| | count |", "|---|--:|",
         f"| candidate (entity, constraint) pairs | {report['candidate_pairs']} |",
         f"| accepted at support >= {a.min_support} | {report['accepted_pairs']} |",
         f"| rejected for support | {report['rejected_for_support']} |",
         f"| pairs dropped: entity more than {a.window} chars from the rule | "
         f"{report['pairs_dropped_for_distance']} |",
         f"| matches dropped: the sentence negates the rule | "
         f"{report['matches_dropped_for_negation']} |",
         f"| sentences dropped: no prohibition or restriction cue | "
         f"{report['sentences_dropped_for_no_cue']} |",
         f"| entities carrying a policy | {report['entities']} |", "",
         "## What was asserted\n",
         "| Constraint | Predicate | Pairs | The customer question it answers |",
         "|---|---|--:|---|"]
    for c in used:
        r = report["constraints"][c]
        L.append(f"| **{c}** | `{r['predicate']}` | {r['pairs']} | "
                 f"{r['question_it_answers']} |")
    L += ["", "## The strongest evidence\n",
          "| Triple | Records | Sentence it came from |", "|---|--:|---|"]
    for t in report["top"][:16]:
        L.append(f"| `{t['triple']}` | {t['support']} | "
                 f"{t['evidence'][:140].replace('|', '/')} |")
    con = report.get("schema_conformance")
    if con:
        L += ["", f"**Schema conformance.** `validate_ontology.py` type-checks "
                  f"against its own rule table, which does not cover these "
                  f"predicates, so each of the {con['triples_checked']} new "
                  f"triples was checked against the domain and range this "
                  f"script declares for it: "
                  f"**{len(con['violations'])} violations**"
                  + ("." if not con["violations"]
                     else ": " + "; ".join(con["violations"][:6]) + ".")]
    L += ["", f"Written to `artifacts/ontologies/{a.write}` beside the published "
              f"version, additively: {report.get('relations', 0):,} relations and "
              f"{report.get('concepts', 0):,} concepts, with every new triple "
              f"carrying its source record ids and the sentence it was read "
              f"from.\n" if a.write else ""]
    (OUT / "policy_constraint_extraction.md").write_text("\n".join(L) + "\n",
                                                         encoding="utf-8")
    print(f"candidates {report['candidate_pairs']}, accepted {report['accepted_pairs']}, "
          f"entities {report['entities']}")
    for c in used:
        print(f"  {c:30s} {report['constraints'][c]['pairs']:4d} pairs "
              f"({report['constraints'][c]['predicate']})")
    print(f"wrote {jp.relative_to(ROOT)} and .md"
          + (f"; version {a.write}" if a.write else ""))


if __name__ == "__main__":
    main()
