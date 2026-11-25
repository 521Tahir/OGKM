"""
Proposal section 11.4.1 - "rule satisfaction and constraint adherence rates",
plus the two automated ontology metrics named there that the Task 2.2 quality
report does not compute: relation redundancy/conflict ratios and hierarchical
BALANCE (depth alone was reported).

Three parts, all offline:

  A  STRUCTURAL ADHERENCE. The seven mined rules (R_mined_009..015) are typed
     patterns - "Accommodation --located_in--> Destination". Each relation
     carrying that predicate either matches the pattern's class signature or
     does not, so adherence is exact, not estimated.

  B  ENFORCEABILITY. The eight seed rules (R001..R008) quantify over attributes
     such as risk severity, traveler type and weather state. A rule whose
     attribute is absent from the ontology cannot be enforced or measured,
     however reasonable it reads. This part reports, per rule, whether the data
     it needs exists - which is a governance finding, not a metric.

  C  ANSWER-LEVEL SATISFACTION. For the seed rules that ARE enforceable, the
     rate at which each variant's answers satisfy them, over the subset of
     benchmark questions that trigger the rule. The detectors are keyword and
     pattern based and are listed in the output so the reader can judge them;
     they are a measurement of rule-shaped behaviour, not a proof of compliance.

  python -m wbs.task2_formalization.rule_compliance
"""
from __future__ import annotations

from wbs.rates import rate

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ONTO = ROOT / "artifacts" / "ontologies"
EVAL = ROOT / "wbs" / "deliverables" / "task4_evaluation"
OUT = ROOT / "wbs" / "deliverables" / "task2_formalization"
VARIANTS = ["V1_llm", "V2_rag", "V3_ontology_only", "V4_ontology_rag",
            "V5_ontology_rag_validated"]

# ── the attributes each seed rule quantifies over ─────────────────────────
# (rule id, needs: ontology attribute or predicate, question trigger, answer test)
HEDGE = re.compile(r"\b(may|might|can vary|varies|typically|generally|usually|"
                   r"recommend(?:ed)? to check|please check|verify|confirm|subject to|"
                   r"approximate|around|about)\b", re.I)
WARN = re.compile(r"\b(caution|careful|warning|warn|advis\w+|avoid|do not|don't|"
                  r"unsafe|danger\w*|risk\w*|safety|emergency|beware|closed)\b", re.I)
REFUSE = re.compile(r"\b(no verified|not available|cannot (?:confirm|verify|provide)|"
                    r"no information|unable to|insufficient)\b", re.I)

SEED_RULES = {
    "R001": dict(needs=["weather state (severe_storm)"],
                 trigger=r"\b(storm|typhoon|rough sea|high wave|bad weather)\b",
                 test="warn", desc="coastal attraction + severe storm -> suppress swimming"),
    "R002": dict(needs=["supports_dietary_need"],
                 trigger=r"\b(halal|kosher|vegan|vegetarian)\b",
                 test="dietary", desc="dietary traveler -> filter dining to matching need"),
    "R003": dict(needs=["activity risk severity", "traveler type"],
                 trigger=r"\b(famil(?:y|ies)|kids|children|toddler|infant)\b",
                 test=None, desc="family travel -> suppress high-risk activities"),
    "R004": dict(needs=["concept confidence"],
                 trigger=None, test="hedge_lowconf",
                 desc="low-confidence concept -> answer cautiously"),
    "R005": dict(needs=["evidence set"], trigger=None, test="refuse_on_empty",
                 desc="no ontology evidence -> do not fabricate"),
    "R006": dict(needs=["has_seasonality", "weather state"],
                 trigger=r"\b(monsoon|rainy season|typhoon season|summer rain)\b",
                 test="warn", desc="monsoon + coastal -> warn on weather-dependent activities"),
    "R007": dict(needs=["has_risk severity"],
                 trigger=r"\b(safe|safety|risk|danger|emergency|hospital|jellyfish|"
                         r"current|tide|hik\w+ alone)\b",
                 test="warn", desc="critical safety risk -> always include warning"),
    "R008": dict(needs=["supports_dietary_need"],
                 trigger=r"\b(halal|kosher|vegan|vegetarian)\b",
                 test="dietary_alt", desc="no matching dietary option -> suggest alternatives"),
}


def gini(xs: list[int]) -> float:
    if not xs or sum(xs) == 0:
        return 0.0
    s = sorted(xs)
    n = len(s)
    cum = sum((i + 1) * v for i, v in enumerate(s))
    return round((2 * cum) / (n * sum(s)) - (n + 1) / n, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--rules-version", default="1.2.0")
    ap.add_argument("--versions", nargs="+", default=["1.2.0", "1.4.0"])
    # Measuring a candidate ontology version must not silently replace the
    # published record of what the deployed one enforces.
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output files, e.g. __1.5.0")
    a = ap.parse_args()

    rules = json.loads((ONTO / a.rules_version / "rules.json").read_text(encoding="utf-8"))
    mined = [r for r in rules if r["rule_id"].startswith("R_mined")]
    seed = [r for r in rules if not r["rule_id"].startswith("R_mined")]
    print(f"{len(rules)} rules from v{a.rules_version}: {len(seed)} seed, "
          f"{len(mined)} mined", flush=True)

    out = {"rules_version": a.rules_version, "n_rules": len(rules),
           "rules_present_by_version": {}, "structural": {}, "enforceability": {},
           "answer_level": {}, "detectors": {
               "hedge": HEDGE.pattern, "warning": WARN.pattern, "refusal": REFUSE.pattern}}

    for v in sorted(p.name for p in ONTO.iterdir() if p.is_dir()):
        out["rules_present_by_version"][v] = (ONTO / v / "rules.json").exists()

    # ── A: structural adherence of mined patterns, per ontology version ───
    pat = re.compile(r"Frequent pattern: (\S+) --(\w+)--> (\S+)")
    sigs = []
    for r in mined:
        m = pat.search(r["description"])
        if m:
            sigs.append((r["rule_id"], m.group(1), m.group(2), m.group(3)))

    for v in a.versions:
        cs = json.loads((ONTO / v / "concepts.json").read_text(encoding="utf-8"))
        rels = json.loads((ONTO / v / "relations.json").read_text(encoding="utf-8"))
        hier_p = ONTO / v / "hierarchy.json"
        hier = json.loads(hier_p.read_text(encoding="utf-8")) if hier_p.exists() else []
        cls = {str(c.get("name", "")).strip().lower(): c.get("concept_class")
               for c in cs if c.get("name")}

        # A mined rule is a FREQUENT PATTERN, not a universal constraint: R_mined_011
        # was mined from 9 observations of Accommodation --located_in--> Destination
        # among 529 located_in edges. Scoring it as "share of all located_in edges
        # that match" would report 1.7% and call the ontology non-compliant, which
        # is a category error. The defensible reading is association-rule
        # confidence - GIVEN the pattern's subject class and predicate, how often
        # does the object class follow - plus its support in the whole graph.
        rows = []
        for rid, sc, pred, oc in sigs:
            app = [r for r in rels
                   if r.get("predicate") == pred
                   and cls.get(str(r.get("subject", "")).strip().lower()) == sc]
            ok = sum(1 for r in app
                     if cls.get(str(r.get("object", "")).strip().lower()) == oc)
            rows.append({"rule_id": rid, "pattern": f"{sc} --{pred}--> {oc}",
                         "antecedent_relations": len(app), "matching": ok,
                         "confidence": round(ok / len(app), 4) if app else None,
                         "support": rate(ok, rels, 6)})

        # redundancy and conflict, the two ratios 11.4.1 names
        trip = Counter()
        pair_preds = defaultdict(set)
        self_loops = 0
        for r in rels:
            s, p, o = (str(r.get(k, "")).strip().lower() for k in ("subject", "predicate", "object"))
            if not (s and p and o):
                continue
            if s == o:
                self_loops += 1
                continue
            trip[(s, p, o)] += 1
            pair_preds[(s, o)].add(p)
        dup = sum(c - 1 for c in trip.values() if c > 1)
        # a conflict is a pair asserted in both directions under a predicate whose
        # inverse is not itself: A located_in B together with B located_in A
        ASYM = {"located_in", "part_of", "is_a", "subclass_of", "reachable_by",
                "served_at", "available_in"}
        conflicts = 0
        for (s, o), preds in pair_preds.items():
            back = pair_preds.get((o, s)) or set()
            if preds & back & ASYM:
                conflicts += 1
        n_rel = len(rels)

        # hierarchical balance
        depths = [len(h.get("hierarchy_path") or []) for h in hier if isinstance(h, dict)]
        children = Counter()
        for r in rels:
            if str(r.get("predicate", "")).lower() in ("is_a", "subclass_of", "instance_of"):
                children[str(r.get("object", "")).strip().lower()] += 1
        class_sizes = Counter(c.get("concept_class") for c in cs if c.get("concept_class"))

        out["structural"][v] = {
            "concepts": len(cs), "relations": len(rels),
            "rules_file_present": (ONTO / v / "rules.json").exists(),
            "mined_rule_adherence": rows,
            "mean_mined_confidence": round(st.mean([r["confidence"] for r in rows
                                                     if r["confidence"] is not None]), 4)
            if any(r["confidence"] is not None for r in rows) else None,
            "mined_rules_with_no_antecedent": sum(1 for r in rows
                                                  if r["confidence"] is None),
            # null, not 0.0, when the version carries no relations: "no
            # redundancy measured" is not "no redundancy" (R15)
            "redundancy_ratio": rate(dup, n_rel),
            "self_loop_ratio": rate(self_loops, n_rel),
            "directional_conflict_pairs": conflicts,
            "conflict_ratio": rate(conflicts, n_rel, 6),
            "hierarchy": {
                "mean_depth": round(st.mean(depths), 4) if depths else None,
                "max_depth": max(depths) if depths else None,
                "depth_histogram": dict(sorted(Counter(depths).items())),
                "mean_branching_factor": round(st.mean(children.values()), 3) if children else None,
                "max_branching_factor": max(children.values()) if children else None,
                "branching_gini": gini(list(children.values())),
                "class_size_gini": gini(list(class_sizes.values())),
                "n_classes": len(class_sizes),
            },
        }
        s_ = out["structural"][v]
        print(f"  v{v}: mined confidence {s_['mean_mined_confidence']} | redundancy "
              f"{s_['redundancy_ratio']} | conflicts {conflicts} | branching gini "
              f"{s_['hierarchy']['branching_gini']}", flush=True)

    # ── B: enforceability of the seed rules against the data that exists ──
    v0 = a.versions[0]
    rels0 = json.loads((ONTO / v0 / "relations.json").read_text(encoding="utf-8"))
    cs0 = json.loads((ONTO / v0 / "concepts.json").read_text(encoding="utf-8"))
    preds0 = {r.get("predicate") for r in rels0}
    has_conf = any(isinstance(c.get("provenance"), dict)
                   and c["provenance"].get("confidence") is not None for c in cs0)
    # Four of these were hardcoded False. Checked against the data on 2026-10-18,
    # one of them was simply wrong: the ontology does carry traveler type - 5 typed
    # concepts and 5/5 correctly typed `suitable_for` triples in the seed, 38
    # concepts and 270/283 in v1.4.0, the version the validator loads. The risk
    # register repeated "no project source supplies ... traveler type", which the
    # data contradicts. The three genuinely absent attributes stay absent, so the
    # enforceable count does not move; what changes is that each verdict is now
    # measured and carries the measurement, and a later version that adds the
    # missing attribute will be noticed instead of being asserted away.
    STORM = re.compile(r"storm|typhoon|rough sea|high wave|gale", re.I)
    SEVERITY = re.compile(r"sever|risk_level|danger_level|\bgrade\b|"
                          r"\b(?:high|medium|low)[_ ]risk\b", re.I)

    def attribute_support(cs, rels) -> dict:
        """Which rule attributes the given ontology version actually carries.

        An attribute counts as available when the class has instances *and* some
        predicate links entities to those instances - a class with no inbound
        predicate is a vocabulary entry, not a queryable attribute.
        """
        cl = {str(c.get("name", "")).strip().lower(): c.get("concept_class")
              for c in cs if c.get("name")}
        by_class = lambda k: [c["name"] for c in cs if c.get("concept_class") == k]

        def linked(klass: str) -> tuple[int, int, str | None]:
            """The predicate that links entities to `klass` most strongly.

            Ranked by the *share* of the predicate's own triples that land on the
            class, not the raw count. This file already holds that a predicate
            existing is not the same as a predicate carrying usable data, and
            measures that with `typed_rate`; the same standard has to apply here or
            an attribute can be declared queryable on the strength of a handful of
            triples inside a predicate that is mostly something else. `has_risk`
            puts 5 of its 72 objects on WeatherCondition and is 27% correctly typed
            overall, which is not a weather attribute.
            """
            best, n_best, tot, share = None, 0, 0, 0.0
            for pr in {r.get("predicate") for r in rels}:
                objs = [str(r.get("object", "")).strip().lower()
                        for r in rels if r.get("predicate") == pr]
                hit = sum(1 for o in objs if cl.get(o) == klass)
                sh = (hit / len(objs)) if objs else 0.0
                if sh > share or (sh == share and hit > n_best):
                    best, n_best, tot, share = pr, hit, len(objs), sh
            return n_best, tot, best

        tt_n, tt_tot, tt_pred = linked("TravelerType")
        wc_n, wc_tot, wc_pred = linked("WeatherCondition")
        # queryable = the linking predicate is mostly about this class, and there
        # is more than a handful of it
        MIN_SHARE, MIN_TRIPLES = 0.5, 5

        def queryable(n, tot):
            return n >= MIN_TRIPLES and tot > 0 and (n / tot) >= MIN_SHARE
        sev = any(SEVERITY.search(str(k)) for c in cs
                  for k in (list(c.keys()) + [c.get("name", "")]))
        return {
            "traveler type": {
                "available": bool(by_class("TravelerType")) and queryable(tt_n, tt_tot),
                "concepts": len(by_class("TravelerType")),
                "linked_by": tt_pred, "linked_triples": tt_n,
                "of_predicate_triples": tt_tot,
                "linked_share": round(tt_n / tt_tot, 4) if tt_tot else None},
            "weather state": {
                # the class exists in v1.4.0 with 8 instances, but the predicate that
                # links to it best is `has_risk`, which puts 5 of 72 objects there.
                # Present in the vocabulary, not queryable as an attribute.
                "available": bool(by_class("WeatherCondition")) and queryable(wc_n, wc_tot),
                "concepts": len(by_class("WeatherCondition")),
                "linked_by": wc_pred, "linked_triples": wc_n,
                "of_predicate_triples": wc_tot,
                "linked_share": round(wc_n / wc_tot, 4) if wc_tot else None},
            "weather state (severe_storm)": {
                "available": any(STORM.search(n) for n in by_class("WeatherCondition")),
                "storm_like_concepts": [n for n in by_class("WeatherCondition")
                                        if STORM.search(n)]},
            "has_risk severity": {"available": sev},
            "activity risk severity": {"available": sev},
        }

    support0 = attribute_support(cs0, rels0)
    v_prod = a.versions[-1]
    try:
        support_prod = attribute_support(
            json.loads((ONTO / v_prod / "concepts.json").read_text(encoding="utf-8")),
            json.loads((ONTO / v_prod / "relations.json").read_text(encoding="utf-8")))
    except Exception:
        support_prod = {}
    # a rule is enforceable if the attribute exists in the seed the rules were
    # written against OR in the version the validator actually loads
    def avail(k: str) -> bool:
        return bool(support0.get(k, {}).get("available")
                    or support_prod.get(k, {}).get("available"))

    AVAILABLE = {
        "supports_dietary_need": "supports_dietary_need" in preds0,
        "has_seasonality": "has_seasonality" in preds0,
        "concept confidence": has_conf,
        "evidence set": True,
        "has_risk severity": avail("has_risk severity"),
        "activity risk severity": avail("activity risk severity"),
        "traveler type": avail("traveler type"),
        "weather state (severe_storm)": avail("weather state (severe_storm)"),
        "weather state": avail("weather state"),
    }
    # A predicate existing is not the same as a predicate carrying usable data.
    # For each predicate a rule depends on, measure the share of its triples whose
    # object is typed as the rule expects; a predicate that fails this is present
    # in name only and a compliance rate computed from it means nothing.
    EXPECTED_OBJ = {"supports_dietary_need": {"DietaryRequirement", "Cuisine",
                                              "Restaurant", "Market"},
                    "has_risk": {"SafetyAdvice", "WeatherCondition", "TravelConstraint"},
                    "has_seasonality": {"Season", "WeatherCondition"}}
    cls_all = {str(c.get("name", "")).strip().lower(): c.get("concept_class")
               for c in cs0 if c.get("name")}
    out["attribute_support"] = {"seed_version": v0, "production_version": v_prod,
                                "seed": support0, "production": support_prod,
                                "min_linked_share": 0.5, "min_linked_triples": 5,
                                "resolved": AVAILABLE}
    typed = {}
    for pred, want in EXPECTED_OBJ.items():
        objs = [str(r.get("object", "")).strip().lower() for r in rels0
                if r.get("predicate") == pred]
        ok = sum(1 for o in objs if cls_all.get(o) in want)
        typed[pred] = {"triples": len(objs), "object_typed_as_expected": ok,
                       "typed_rate": round(ok / len(objs), 4) if objs else None,
                       "expected_object_classes": sorted(want),
                       "observed_object_classes": sorted(
                           {str(cls_all.get(o)) for o in objs})[:12]}
        r_ = typed[pred]
        print(f"  predicate {pred}: {ok}/{len(objs)} triples have an expected object "
              f"class ({(r_['typed_rate'] or 0)*100:.0f}%)", flush=True)
    out["predicate_data_quality"] = typed
    USABLE_MIN = 0.50          # below half correctly typed, the layer is noise

    for rid, spec in SEED_RULES.items():
        missing = [n for n in spec["needs"] if not AVAILABLE.get(n, False)]
        # a predicate the rule needs whose data is dominated by mistyped objects
        weak = [pr for pr in spec["needs"]
                if pr in typed and (typed[pr]["typed_rate"] or 0) < USABLE_MIN]
        out["enforceability"][rid] = {
            "description": spec["desc"], "needs": spec["needs"],
            "missing_from_ontology": missing,
            "present_but_noise_dominated": weak,
            "enforceable": not missing,
            "measurable_here": (spec["test"] is not None and not missing and not weak),
            "why_not_measurable": ("attribute absent: " + ", ".join(missing)) if missing
            else ("predicate present but under "
                  f"{int(USABLE_MIN*100)}% of its triples are correctly typed: "
                  + ", ".join(weak)) if weak
            else ("no benchmark question triggers it" if spec["test"] else "no detector"),
        }
    n_enf = sum(1 for x in out["enforceability"].values() if x["enforceable"])
    print(f"\nseed rules enforceable with the data that exists: {n_enf}/{len(SEED_RULES)}",
          flush=True)
    for rid, x in out["enforceability"].items():
        if not x["enforceable"]:
            print(f"  {rid} unenforceable - ontology has no "
                  f"{', '.join(x['missing_from_ontology'])}", flush=True)

    # ── C: answer-level satisfaction for the measurable seed rules ────────
    # The dietary words themselves must not count as evidence of compliance: a
    # question asking about vegan food makes every answer that repeats the word
    # "vegan" look compliant. Only concrete venues and dishes the ontology records
    # as supporting a dietary need are admissible.
    GENERIC_DIET = {"halal", "kosher", "vegan", "vegetarian", "dietary",
                    "dietary requirement", "dietary needs"}
    diet_objs = {str(r.get("object", "")).strip().lower() for r in rels0
                 if r.get("predicate") == "supports_dietary_need"} - GENERIC_DIET
    lowconf = {str(c.get("name", "")).strip().lower() for c in cs0
               if isinstance(c.get("provenance"), dict)
               and (c["provenance"].get("confidence") or 1.0) < 0.5}
    print(f"{len(diet_objs)} dietary objects, {len(lowconf)} low-confidence concepts",
          flush=True)

    answers = {}
    for v in VARIANTS:
        p = EVAL / f"variant__{a.model}__{v}.jsonl"
        if p.exists():
            answers[v] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if not answers:
        print("no variant answers found; skipping part C", flush=True)
    else:
        for rid, spec in SEED_RULES.items():
            if not out["enforceability"][rid]["measurable_here"]:
                continue
            trig = re.compile(spec["trigger"], re.I) if spec["trigger"] else None
            per_variant = {}
            for v, rows in answers.items():
                hits, sat = 0, 0
                for r in rows:
                    q, ans = r.get("Input") or "", r.get("answer") or ""
                    if trig is not None and not trig.search(q):
                        continue
                    if spec["test"] == "hedge_lowconf":
                        low = ans.lower()
                        if not any(c in low for c in lowconf):
                            continue
                    if spec["test"] == "refuse_on_empty":
                        # R005's precondition is an EMPTY evidence set. The V3
                        # builder falls back to a default class mix, so it returns
                        # facts for every benchmark question and the precondition
                        # never arises here. Recorded as not-triggered rather than
                        # satisfied, because an untriggered rule is not a pass.
                        continue
                    hits += 1
                    if spec["test"] in ("warn",):
                        sat += 1 if WARN.search(ans) else 0
                    elif spec["test"] == "hedge_lowconf":
                        sat += 1 if HEDGE.search(ans) else 0
                    elif spec["test"] == "dietary":
                        low = ans.lower()
                        sat += 1 if any(d in low for d in diet_objs) else 0
                    elif spec["test"] == "dietary_alt":
                        low = ans.lower()
                        sat += 1 if (any(d in low for d in diet_objs)
                                     or re.search(r"\b(alternative|instead|nearby|market|"
                                                  r"self-cater|ask)\b", low)) else 0
                per_variant[v] = {"triggered": hits,
                                  "satisfied": sat,
                                  "rate": round(sat / hits, 4) if hits else None}
            # Self-check: V1 receives no retrieval of any kind. If it satisfies the
            # detector as often as the best grounded arm, the detector is measuring
            # topical vocabulary rather than rule compliance, and the rate must not
            # be read as a compliance figure.
            v1 = (per_variant.get("V1_llm") or {}).get("rate")
            grounded = [x["rate"] for v, x in per_variant.items()
                        if v != "V1_llm" and x["rate"] is not None]
            degenerate = (v1 is not None and grounded and v1 >= max(grounded))
            out["answer_level"][rid] = {"description": spec["desc"],
                                        "trigger": spec["trigger"],
                                        "test": spec["test"], "by_variant": per_variant,
                                        "detector_degenerate": bool(degenerate),
                                        "detector_note":
                                        "V1 (no retrieval) matches or beats every "
                                        "grounded arm, so this rate reflects topical "
                                        "vocabulary, not rule compliance"
                                        if degenerate else ""}
            n_t = max((x["triggered"] for x in per_variant.values()), default=0)
            line = "  ".join(f"{v.split('_')[0]} {x['rate']*100:.0f}%" if x["rate"] is not None
                             else f"{v.split('_')[0]} —" for v, x in per_variant.items())
            flag = "  [DEGENERATE: V1 >= grounded arms]" if out["answer_level"][rid][
                "detector_degenerate"] else ""
            print(f"  {rid} (n={n_t}): {line}{flag}", flush=True)

    tag = a.out_tag
    if a.versions != ap.get_default("versions") and not tag:
        sys.exit(f"refusing to overwrite the published rule record with "
                 f"{a.versions}: pass --out-tag (e.g. --out-tag __{a.versions[-1]})")
    p = OUT / f"rule_compliance{tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ── markdown deliverable ──────────────────────────────────────────────
    L = ["# Rule satisfaction and constraint adherence (proposal 11.4.1)\n"]
    missing_versions = [v for v, ok in out["rules_present_by_version"].items() if not ok]
    L.append(f"The ontology carries {len(rules)} rules - {len(seed)} seed rules written "
             f"from domain knowledge and {len(mined)} patterns mined from the extracted "
             f"graph - serialized in v{a.rules_version}. They are **absent from "
             f"{', '.join('v' + v for v in missing_versions)}**"
             + (" — and v1.4.0 is the version the answer validator loads, so the "
                "production ontology carries no rule layer at all."
                if "1.4.0" in missing_versions else ".") + "\n")

    L.append("## Mined patterns are descriptive, not prescriptive\n")
    L.append("A mined rule records that a pattern occurred often, not that it must "
             "always hold: R_mined_011 comes from 9 observations of "
             "`Accommodation --located_in--> Destination`. Scoring the graph against "
             "it as a constraint would report near-total non-compliance and mean "
             "nothing. Association-rule confidence is the defensible reading - given "
             "the antecedent class and predicate, how often the consequent class "
             "follows.\n")
    for v in a.versions:
        s_ = out["structural"][v]
        L.append(f"**v{v}** — {s_['concepts']:,} concepts, {s_['relations']:,} "
                 f"relations, rules file "
                 f"{'present' if s_['rules_file_present'] else '**absent**'}. "
                 f"Mean pattern confidence **{s_['mean_mined_confidence']}**.\n")
        L.append("| Rule | Pattern | Antecedent edges | Consequent holds | Confidence | Support |")
        L.append("|---|---|--:|--:|--:|--:|")
        for r in s_["mined_rule_adherence"]:
            c = f"{r['confidence']:.3f}" if r["confidence"] is not None else "—"
            L.append(f"| {r['rule_id']} | `{r['pattern']}` | "
                     f"{r['antecedent_relations']:,} | {r['matching']:,} | **{c}** | "
                     f"{r['support']:.5f} |")
        thin = [r["rule_id"] for r in s_["mined_rule_adherence"]
                if (r["antecedent_relations"] or 0) < 10]
        if thin:
            L.append(f"{', '.join(thin)} have fewer than 10 antecedent edges in this "
                     f"version, so their confidence is a ratio of single digits and "
                     f"should not be read as a property of the ontology. The support "
                     f"column is there to make that visible: no pattern reaches "
                     f"{max(r['support'] for r in s_['mined_rule_adherence']):.2%} of "
                     f"the graph.\n")
        L.append("")
    L.append("## The two ratios 11.4.1 names, and hierarchical balance\n")
    L.append("| Metric | " + " | ".join(f"v{v}" for v in a.versions) + " |")
    L.append("|---|" + "--:|" * len(a.versions))
    def grab(path):
        vals = []
        for v in a.versions:
            o = out["structural"][v]
            for k in path:
                o = o[k] if isinstance(o, dict) else None
            vals.append("—" if o is None else (f"{o:,}" if isinstance(o, int) else str(o)))
        return vals
    for label, path in [("Relation redundancy ratio", ["redundancy_ratio"]),
                        ("Self-loop ratio", ["self_loop_ratio"]),
                        ("Directional conflict pairs", ["directional_conflict_pairs"]),
                        ("Conflict ratio", ["conflict_ratio"]),
                        ("Mean hierarchy depth", ["hierarchy", "mean_depth"]),
                        ("Max hierarchy depth", ["hierarchy", "max_depth"]),
                        ("Mean branching factor", ["hierarchy", "mean_branching_factor"]),
                        ("Max branching factor", ["hierarchy", "max_branching_factor"]),
                        ("Branching Gini (0 = balanced)", ["hierarchy", "branching_gini"]),
                        ("Class-size Gini", ["hierarchy", "class_size_gini"]),
                        ("Concept classes", ["hierarchy", "n_classes"])]:
        L.append(f"| {label} | " + " | ".join(grab(path)) + " |")
    v_last = a.versions[-1]
    hb = out["structural"][v_last]["hierarchy"]
    L.append(f"\nDepth alone was already reported; balance was not. A branching Gini of "
             f"**{hb['branching_gini']}** on v{v_last} says the taxonomy is far from "
             f"balanced: a few parents carry most of the children (max branching "
             f"{hb['max_branching_factor']:,} against a mean of "
             f"{hb['mean_branching_factor']}). The directional conflicts - a pair "
             f"asserted both ways under an asymmetric predicate, such as A located_in B "
             f"together with B located_in A - number "
             f"{out['structural'][v_last]['directional_conflict_pairs']:,} and are the "
             f"same extraction noise the L2 validation layer reports as warnings.\n")

    L.append("## Can the seed rules be enforced at all?\n")
    L.append("A rule that quantifies over an attribute the ontology does not record "
             "cannot be enforced, measured, or claimed - however reasonable it reads. "
             "This is the finding, not a metric.\n")
    L.append("| Rule | What it says | Needs | Enforceable |")
    L.append("|---|---|---|---|")
    for rid, x in out["enforceability"].items():
        need = ", ".join(f"`{n}`" for n in x["needs"])
        ok = "✅ yes" if x["enforceable"] else f"❌ no — missing {', '.join(x['missing_from_ontology'])}"
        L.append(f"| {rid} | {x['description']} | {need} | {ok} |")
    n_enf = sum(1 for x in out["enforceability"].values() if x["enforceable"])
    n_meas = sum(1 for x in out["enforceability"].values() if x["measurable_here"])
    # the list of blocking attributes is read off the verdicts, because asserting
    # it is how "traveler type" survived here for weeks while `suitable_for`
    # carried it on 270 of its 283 triples
    blocked = {rid: x["missing_from_ontology"]
               for rid, x in out["enforceability"].items()
               if not x["enforceable"]}
    attrs_blocking = sorted({m for ms in blocked.values() for m in ms})
    L.append(f"\n**{n_enf} of {len(SEED_RULES)} seed rules are enforceable against the "
             f"data that exists.** The {len(blocked)} that are not are blocked by "
             + ", ".join(f"`{x}`" for x in attrs_blocking)
             + f": {', '.join(sorted(blocked))} are unevaluable, so they are "
               f"statements of intent and the project should not report a compliance "
               f"figure for them. Each verdict is measured per ontology version from "
               f"class instances and the share of the linking predicate's own triples "
               f"that reach them, not asserted — an attribute is counted as carried "
               f"only when some predicate is mostly about it (≥"
             + f"{int(out['attribute_support']['min_linked_share']*100)}% of its own "
               f"triples, at least "
               f"{out['attribute_support']['min_linked_triples']}). "
               f"**Traveler type is not among the blockers**, though it was listed as "
               f"one until 2026-10-18: `suitable_for` carries it on "
             + (lambda e: f"{e['linked_triples']} of {e['of_predicate_triples']}")(
                   out["attribute_support"]["production"]["traveler type"])
             + " triples in the production version, and the absence was a hardcoded "
               "`False` in the availability map rather than a finding.\n")
    if out.get("predicate_data_quality"):
        L.append("A predicate existing is not the same as a predicate carrying usable "
                 "data, so each one a rule depends on was checked for whether its "
                 "objects are typed as the rule expects:\n")
        L.append("| Predicate | Triples | Object typed as expected | Observed object classes |")
        L.append("|---|--:|--:|---|")
        for pr, q in out["predicate_data_quality"].items():
            L.append(f"| `{pr}` | {q['triples']} | "
                     f"{q['object_typed_as_expected']} "
                     f"({(q['typed_rate'] or 0)*100:.0f}%) | "
                     + ", ".join(c for c in q["observed_object_classes"] if c != "None")[:90]
                     + " |")
        L.append("")
    if out["answer_level"]:
        L.append("## Answer-level satisfaction, and why most of it is not reportable\n")
        L.append("| Rule | Triggered | " + " | ".join(
            v.replace("_", " ") for v in VARIANTS) + " |")
        L.append("|---|--:|" + "--:|" * len(VARIANTS))
        for rid, x in out["answer_level"].items():
            bv = x["by_variant"]
            n_t = max((y["triggered"] for y in bv.values()), default=0)
            cells = []
            for v in VARIANTS:
                y = bv.get(v) or {}
                cells.append(f"{y['rate']*100:.0f}%" if y.get("rate") is not None else "—")
            L.append(f"| {rid} | {n_t} | " + " | ".join(cells) + " |")
        degen = [rid for rid, x in out["answer_level"].items() if x["detector_degenerate"]]
        if degen:
            L.append(f"\n{', '.join(degen)} are marked **degenerate**: V1, which "
                     f"receives no retrieval of any kind, satisfies the detector as "
                     f"often as every grounded arm. A detector a context-free model "
                     f"passes is measuring topical vocabulary, not rule compliance, so "
                     f"these rates are reported and then discarded rather than quoted "
                     f"as evidence. The underlying cause is visible in the table above: "
                     f"the dietary layer is {out['predicate_data_quality']['supports_dietary_need']['triples']} "
                     f"triples whose admissible objects are generic categories that any "
                     f"tourism answer mentions.\n")
        L.append("Rules whose trigger never fires are reported as not-triggered, never "
                 "as satisfied: R004 needs a concept with confidence below 0.5 and the "
                 "ontology has none, and R005 needs an empty evidence set, which the "
                 "V3 builder's default class fallback prevents. An untriggered rule is "
                 "not a passed rule.\n")
    L.append("## Consequence for the project\n")
    L.append(f"Of the {len(rules)} rules the proposal's section 5.6 pipeline produced, "
             f"{len(mined)} are descriptive patterns correctly read as association "
             f"rules, {len(SEED_RULES) - n_enf} cannot be enforced because the ontology "
             f"lacks their attributes, and of the remainder none yields a "
             f"non-degenerate answer-level compliance rate on this benchmark. The "
             f"honest reading of proposal 11.4.1's \"rule satisfaction and constraint "
             f"adherence rates\" is that the constraint layer that actually governs "
             f"generation is the Task 3.2.2 validator - grounding, unsupportable-claim "
             f"and relation-contradiction checks, whose firing rates ARE measured on "
             f"real output - and not these 15 rule records.\n")
    (OUT / f"rule_compliance{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {p.name}, rule_compliance{tag}.md")


if __name__ == "__main__":
    main()
